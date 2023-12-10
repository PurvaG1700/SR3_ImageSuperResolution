import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob

from ignite.metrics import FID, InceptionScore
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    small_names = list(glob.glob('{}/*_lr.png'.format(args.path)))
    real_names.sort()
    fake_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_mse = 0.0
    idx = 0

    # inception = InceptionScore()
    # fid = FID()

    for rname, fname, sname in zip(real_names, fake_names, small_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_sr")[0]
        # assert ridx == fidx, 'Image ridx:{}!=fidx:{}'.format(ridx, fidx)

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        lr_img = np.array(Image.open(sname))

        # fid.update((sr_img, hr_img))
        # inception.update((sr_img, hr_img))

        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)

        resized_sr = np.array(Image.open(fname).resize(
            (16, 16), resample=Image.BICUBIC))
        mse = ((resized_sr - lr_img)**2).mean()
        avg_psnr += psnr
        avg_ssim += ssim
        avg_mse += mse
        if idx % 20 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_mse = avg_mse / idx

    # print('# Validation # IS: {:.4e}'.format(inception.compute()))
    # print('# Validation # FID: {:.4e}'.format(fid.compute()))

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    print('# Validation # Cosistency: {:.4e}'.format(avg_mse))
