import os
import time
from datetime import datetime
import sys
import random
import argparse
from net.MDJCM import MDJCM_Model
import torch.optim as optim
from utils import *
from data.datasets import get_loader, get_test_loader
from config import config
from torchvision import transforms
from PIL import Image


def train_one_epoch(epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger, modulation_order=None, snr=None):
    global global_step
    net.train()
    elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs,MSE_cecd = [AverageMeter() for _ in range(8)]
    metrics = [elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs,MSE_cecd]
    for batch_idx, input_image in enumerate(train_loader):
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()

        start_time = time.time() 
        input_image = input_image.to(device)
        global_step += 1
        mse_loss_ntc, bpp_y, bpp_z, mse_loss_MDJCM, cbr_y, x_hat_ntc, x_hat_MDJCM,mse_cecd = net(input_image, modulation_order, snr)
        if config.use_side_info:
            cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
            loss = mse_loss_MDJCM + mse_loss_ntc + config.train_lambda * (bpp_y * config.eta + cbr_z)
            cbrs.update(cbr_y + cbr_z)
        else:
            # add ntc_loss to improve the training convergence stability
            ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
            loss = ntc_loss + mse_loss_MDJCM
            cbrs.update(cbr_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer_G.step()

        aux_loss = net.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        bppys.update(bpp_y.item())
        bppzs.update(bpp_z.item())

        psnr_jscc = 10 * (torch.log(255. * 255. / mse_loss_MDJCM) / np.log(10))
        psnr_jsccs.update(psnr_jscc.item())
        psnr = 10 * (torch.log(255. * 255. / mse_loss_ntc) / np.log(10))
        psnrs.update(psnr.item())
        MSE_cecd.update(mse_cecd)
        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR_JSCC {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'MSE_cecd {MSE_cecd.val:.4f} ({MSE_cecd.avg:.4f})',
                f'PSNR_NTC {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'Epoch {epoch}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()


def test(net, test_loader, logger, modulation_order=None, snr=None):
    with torch.no_grad():
        net.eval()
        elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs,MSE_cecd = [AverageMeter() for _ in range(8)]
        PSNR_list = []
        CBR_list = []
        for batch_idx, input_image in enumerate(test_loader):
            start_time = time.time()
            input_image = input_image.cuda()
            
            target_size = (input_image.shape[2]-input_image.shape[2]%128,  input_image.shape[3]-input_image.shape[3]%128)
            center_crop = transforms.CenterCrop(target_size)
            input_image = center_crop(input_image)
            
            cbr_sideinfo = np.log2(16) / (16 * 16 * 3) / np.log2(
        1 + 10 ** (net.channel.chan_param / 10))   
            
            mse_loss_ntc, bpp_y, bpp_z, mse_loss_MDJCM, cbr_y, x_hat_ntc, x_hat_MDJCM , mse_cecd= net(input_image, modulation_order, snr)
            
            if config.use_side_info:
                cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                MDJCM_loss = mse_loss_MDJCM + bpp_y * config.eta + cbr_z
                loss = ntc_loss + MDJCM_loss
                cbrs.update(cbr_y + cbr_z)
                
            else:
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                loss = ntc_loss + mse_loss_MDJCM + 10*mse_cecd
                cbrs.update(cbr_y)
            losses.update(loss.item())
            bppys.update(bpp_y)
            bppzs.update(bpp_z)
            elapsed.update(time.time() - start_time)

            psnr_jscc = CalcuPSNR_int(input_image, x_hat_MDJCM).mean()
            psnr_jsccs.update(psnr_jscc)
            psnr = CalcuPSNR_int(input_image, x_hat_ntc).mean()
            psnrs.update(psnr)
            MSE_cecd.update(mse_cecd)
            log = (' | '.join([
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'MSE {mse_loss_MDJCM:.3f} ({mse_loss_ntc:.3f})',
                f'Time {elapsed.val:.2f}',
                f'PSNR1 {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})',
                f'CBR {cbrs.val+cbr_sideinfo:.4f} ({cbrs.avg + cbr_sideinfo:.4f})',
                f'MSE_cecd {MSE_cecd.val:.4f} ({MSE_cecd.avg:.4f})',
                f'PSNR2 {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
            ]))
            logger.info(log)
            PSNR_list.append(psnr_jscc)
            CBR_list.append(cbr_y)

    # Here, the channel bandwidth cost of side info \bar{k} is transmitted by a capacity-achieving channel code. Note
    # that, the side info should be transmitted through entropy coding and channel coding, which will be addressed in
    # future releases.

    # capacity-achieving channel code
    cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16 * 16 * 3) / np.log2(
        1 + 10 ** (net.channel.chan_param / 10))

    logger.info(f'Finish test! Average PSNR={psnr_jsccs.avg:.4f}dB, CBR={cbrs.avg + cbr_sideinfo:.4f}')
    return losses.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training/testing script.")
    parser.add_argument(
        "-p",
        "--phase",
        default='train',  # train
        type=str,
        help="Train or Test",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=150,
        type=int,
        help="Number of epochs (default: %(default)s)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=2,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=1024, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument(
        '--save_log', action='store_true', default=True, help='Save log to disk'
    )
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def load_checkpoint(model,resume):
    checkpoint = torch.load(resume, map_location='cuda')
    print(resume)
    print("Load ckpt from the place")
    checkpoint_model = checkpoint
    state_dict = model.state_dict()
    all_keys = list(checkpoint.keys())
    for k in all_keys:
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    return checkpoint_model


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def main(argv):
    args = parse_args(argv)

    if args.seed is not None: 
        torch.manual_seed(args.seed) 
        random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.device = device

    workdir, logger = logger_configuration(args.name, phase=args.phase, save_log=args.save_log)
    config.logger = logger
    logger.info(config.__dict__)

    net = MDJCM_Model(config).cuda()
    model_path = args.checkpoint
    if model_path:
        load_weights(net, model_path)
    
    
    logger.info('======Checkpoint %s ======' % args.checkpoint)

    modulation_order = None
    snr = None
    if args.phase == 'test':
        if config.modulation:
                modulation_order = config.modulation_order
        if config.channel_adaptive:
                snr = config.channel['chan_param']
        test_loader = get_test_loader(config)
        test(net, test_loader, logger, modulation_order, snr) 
        # save_model(net, save_path=workdir + '/models/EP{}.model'.format(1))
    elif args.phase == 'train':
        train_loader, test_loader = get_loader(config)
        global global_step
        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        # optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, G_params), lr=config.lr)
        aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[4000, 4500], gamma=0.1)
        tot_epoch = 5000
        global_step = 0
        best_loss = float("inf")
        steps_epoch = global_step // train_loader.__len__()

        
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")

            if config.modulation:
                modulation_order = random.choices([4, 16, 64, 256, 1024])[0]
            if config.channel_adaptive:
                snr = np.random.rand()*20
            train_one_epoch(epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger, modulation_order, snr)
            lr_scheduler.step()

            loss = test(net, test_loader, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_model(net, save_path=workdir + '/models/EP{}_best_loss.model'.format(epoch + 1))
            save_model(net, save_path=workdir + '/models/EP{}.model'.format(epoch + 1))


if __name__ == '__main__':
    main(sys.argv[1:])
