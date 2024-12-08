import torch
import torch.nn.functional as F
from cnn_utils import SameShapeConv1d


class ENCBase(torch.nn.Module):
    def __init__(self, args):
        super(ENCBase, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        # self.args = args
        self.reset_precomp()

    # def set_parallel(self):
    #     pass

    def set_precomp(self, mean_scalar, std_scalar):
        self.mean_scalar = mean_scalar.to(self.this_device)
        self.std_scalar  = std_scalar.to(self.this_device)

    # not tested yet
    def reset_precomp(self):
        self.mean_scalar = torch.zeros(1).type(torch.FloatTensor).to(self.this_device)
        self.std_scalar  = torch.ones(1).type(torch.FloatTensor).to(self.this_device)
        self.num_test_block= 0.0

    def enc_act(self, inputs):
        return F.elu(inputs)
    
class ENC_interCNN(ENCBase): 
    # def __init__(self, args, p_array):
    def __init__(self, args):
        # turbofy only for code rate 1/3
        super(ENC_interCNN, self).__init__(args)  
        self.args             = args

        # Encoder
        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

    def forward(self, inputs): 

        # inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys)) 

        x_sys_int  = inputs 
        x_p2       = self.enc_cnn_2(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_2(x_p2))

        x_tx       = torch.cat([x_sys, x_p2], dim = 2)
        return x_tx
    

class DEC_interCNN(torch.nn.Module):
    def __init__(self, args):
        super(DEC_interCNN, self).__init__()
        self.args = args

        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        # self.interleaver          = Interleaver(args, p_array)
        # self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            if idx==0:
                self.dec1_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=1,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )
            else:
                self.dec1_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )
            if idx == args.num_iteration -1:
                self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, 1))
            else:
                self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))
            # self.dec1_outputs.append(torch.nn.Linear(args.num_iter_ft, 1))

    def forward(self, received):
        batch_size,block_len,_ = received.shape
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Decoder
        r_sys       = received.view((batch_size, block_len, 1))

        for idx in range(self.args.num_iteration):
            # x_this_dec = torch.cat([r_sys,r_par_deint, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](r_sys)
            x_plr      = self.dec1_outputs[idx](x_dec)
            r_sys = x_plr

        # last round
        # final = self.dec1_outputs[self.args.num_iteration-1](r_sys)
        final = r_sys
        return final 