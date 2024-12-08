import numpy as np
import torch
import math
import torch.nn as nn
from loss.distortion import Distortion
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import ste_round
from layer.layers import Mlp
from layer.analysis_transform import AnalysisTransform
from layer.synthesis_transform import SynthesisTransform
from layer.jscc_encoder import JSCCEncoder
from layer.jscc_decoder import JSCCDecoder
from utils import BCHW2BLN, BLN2BCHW, qam_mapper
from channel.channel import Channel 

import torch.nn.functional as F
from cnn_utils import SameShapeConv1d
from interleavers import ENC_interCNN, DEC_interCNN

class NTC_Hyperprior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ga = AnalysisTransform(**config.ga_kwargs)
        self.gs = SynthesisTransform(**config.gs_kwargs)
        self.ha = nn.Sequential(
            nn.Conv2d(256, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(192, 192, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(192, 192, 5, stride=2, padding=2),
        )

        self.hs = nn.Sequential(
            nn.ConvTranspose2d(192, 256, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 512, 3, stride=1, padding=1)
        )
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.distortion = Distortion(config)
        self.H = self.W = 0

    def update_resolution(self, H, W):
        if H != self.H or W != self.W:
            self.ga.update_resolution(H, W)
            self.gs.update_resolution(H // 16, W // 16)
            self.H = H
            self.W = W

    def forward(self, input_image, require_probs=False):
        B, C, H, W = input_image.shape
        self.update_resolution(H, W)
        y = self.ga(input_image)
        z = self.ha(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        gaussian_params = self.hs(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y - means_hat) + means_hat
        x_hat = self.gs(y_hat)
        mse_loss = self.distortion(input_image, x_hat)
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        if require_probs:
            return mse_loss, bpp_y, bpp_z, x_hat, y, y_likelihoods, scales_hat, means_hat
        else:
            return mse_loss, bpp_y, bpp_z, x_hat

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss


class MDJCM_Model(NTC_Hyperprior):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.channel = Channel(config)
        self.fe = JSCCEncoder(**config.fe_kwargs)
        self.fd = JSCCDecoder(**config.fd_kwargs)
        if config.use_side_info:
            embed_dim = config.fe_kwargs['embed_dim']
            self.hyprior_refinement = Mlp(embed_dim * 3, embed_dim * 6, embed_dim)
        self.eta = config.eta
        self.mse = nn.MSELoss()
 
        self.block_len=config.block_len
        self.CD = DEC_interCNN(config)


    def feature_probs_based_Gaussian(self, feature, mean, sigma):
        sigma = sigma.clamp(1e-10, 1e10) if sigma.dtype == torch.float32 else sigma.clamp(1e-10, 1e4)
        gaussian = torch.distributions.normal.Normal(mean, sigma)
        prob = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        likelihoods = torch.clamp(prob, 1e-10, 1e10)  # B C H W
        return likelihoods

    def update_resolution(self, H, W):
        # Update attention mask for W-MSA and SW-MSA
        if H != self.H or W != self.W:
            self.ga.update_resolution(H, W)
            self.fe.update_resolution(H // 16, W // 16)
            self.gs.update_resolution(H // 16, W // 16)
            self.fd.update_resolution(H // 16, W // 16)
            self.H = H
            self.W = W

    def forward(self, input_image, modulation_order, snr, **kwargs):
        device = input_image.device 
        B, C, H, W = input_image.shape
        num_pixels = H * W * 3
        self.update_resolution(H, W)

        # NTC forward
        mse_loss_ntc, bpp_y, bpp_z, x_hat_ntc, y, y_likelihoods, scales_hat, means_hat = \
            self.forward_NTC(input_image, require_probs=True)
        y_likelihoods = self.feature_probs_based_Gaussian(y, means_hat, scales_hat)

        # DJSCC forward
        s_masked, mask_BCHW, indexes = self.fe(y, y_likelihoods.detach(), eta=self.eta, modulation_order = modulation_order, snr = snr)

        # Pass through the channel.
        # ****************************** fe ****************************** #
        mask_BCHW = mask_BCHW.byte()
        mask_BCHW = mask_BCHW.bool()
        channel_input = torch.masked_select(s_masked, mask_BCHW)
        # ************************** Modulation ************************** #
        if self.config.modulation:
            # print('modualtion')
            # Power normalization
            power = 1
            avg_pwr = torch.mean(channel_input ** 2)
            channel_input = np.sqrt(power) * channel_input / torch.sqrt(avg_pwr * 2)
            d = math.sqrt(3*power/(2*modulation_order-2))
            channel_input = torch.clamp(channel_input,(1-math.sqrt(modulation_order))*d,(math.sqrt(modulation_order)-1)*d).to(device)
            if self.training:
                u1 = (torch.rand(channel_input.shape).to(device)-0.5)*2*d
                channel_input += u1
            else:
                channel_input = qam_mapper(channel_input, modulation_order, d)     
        # *********************************************************************#

        # **************************** channel *************************** #
        channel_input_copy = channel_input
        channel_output, channel_usage = self.channel.forward(channel_input, snr)
        # ************************* Demodulation ************************* #
        if self.config.modulation:
            # print('demodualtion')
            if self.training:
                u2 = (torch.rand(channel_output.shape).to(device)-0.5)*2*d
                channel_output += u2
            else:
                channel_output = qam_mapper(channel_output, modulation_order, d)     
        # ****************************** CD ****************************** #
        cout = torch.zeros_like(s_masked)
        cout[mask_BCHW] = channel_output
        merging = ()   
        mask_new = ()
        for i in range(B):
            tensor_i = torch.masked_select(cout[i], mask_BCHW[i])
            padding_size = self.block_len-len(tensor_i)%self.block_len
            padding_tensor = torch.zeros(padding_size).to(device)
            padded_tensor = torch.cat((tensor_i, padding_tensor))
            padded_tensor = padded_tensor.reshape(-1,self.block_len,1)
            mask = torch.ones_like(tensor_i)
            mask = torch.cat((mask, padding_tensor))
            mask = mask.reshape(-1,self.block_len,1)
            merging = merging+ (padded_tensor,)
            mask_new = mask_new + (mask,)
        merged_tensor = torch.cat(merging, dim=0)
        mask_new = torch.cat(mask_new,dim=0)
        channel_output = self.CD(merged_tensor)
        
        mask_new = mask_new.bool()
        channel_output = torch.masked_select(channel_output, mask_new)
        
        mse_cecd = self.mse(channel_input_copy,channel_output)
        
        # ****************************** fd ****************************** #
        s_hat = torch.zeros_like(s_masked)
        s_hat[mask_BCHW] = channel_output
        cbr_y = channel_usage / num_pixels
        y_hat = self.fd(s_hat, indexes, modulation_order = modulation_order, snr = snr) 
        
        
        # hyperprior-aided decoder refinement (optional)
        if self.config.use_side_info:
            y_combine = torch.cat([BCHW2BLN(y_hat), BCHW2BLN(means_hat), BCHW2BLN(scales_hat)], dim=-1)
            y_hat = BLN2BCHW(BCHW2BLN(y_hat) + self.hyprior_refinement(y_combine), H // 16, W // 16)

        x_hat_MDJCM = self.gs(y_hat).clip(0, 1)
        mse_loss_MDJCM = self.distortion(input_image, x_hat_MDJCM)

        return mse_loss_ntc, bpp_y, bpp_z, mse_loss_MDJCM, cbr_y, x_hat_ntc, x_hat_MDJCM,mse_cecd

    def forward_NTC(self, input_image, **kwargs):
        return super(MDJCM_Hyperprior, self).forward(input_image, **kwargs)
