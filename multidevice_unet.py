import os
from pathlib import Path
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torchvision.utils import save_image

import torch.multiprocessing as mp

import sys

from model.utils import RandomVerticalFlipWithState, RandomHorizontalFlipWithState, read_image, calculate_psnr, calculate_ssim, add_noise, add_mask, average_masked_image
import torch
import torch.nn.functional as F
from model.model import PartialConvUnet
import signal
import multiprocessing
import argparse

def train_model(Gamma, step_size, path, file_name, model, optimizer, noise_lvl, mask_ratio, num_epochs, device='cuda:1', gray=False, exp = 'exp'):
    os.makedirs(f'Dropout/{exp}', exist_ok=True)
    sys.stdout = open(f'Dropout/{exp}/{noise_lvl}_{file_name[:-4]}.txt', 'w', buffering=1)
    print('------------------------------------------------------------')
    print('noise level: {}'.format(noise_lvl))
    print('file name: {}'.format(file_name))
    print('mask ratio: {}'.format(mask_ratio))

    if file_name == 'image_Baboon512rgb.png' or 'kodim01.png' :
        if noise_lvl < 75:
            pass
        elif noise_lvl == 75:
            num_epochs = 150000
        else:
            num_epochs = 50000
    else:
        if noise_lvl == 25:
            pass
        elif noise_lvl == 50:
            num_epochs = 150000
        elif noise_lvl == 75:
            num_epochs = 100000
        else:
            num_epochs = 50000
            
        if file_name == 'image_House256rgb.png':
            if noise_lvl == 25:
                num_epochs = 70000
            elif noise_lvl == 50:
                num_epochs = 50000
            else:
                num_epochs = 30000
      
    print('num epochs: {}'.format(num_epochs))
    print('gray: {}'.format(gray))
    print('------------------------------------------------------------')
    file_path = os.path.join(path, file_name)
    image = read_image(file_path, device=device, gray=gray)
    noisy_image = add_noise(image, noise_lvl, device=device)
    vertical_flip = RandomVerticalFlipWithState(1/3)
    horizontal_flip = RandomHorizontalFlipWithState(1/3)
    transform = transforms.Compose([
        vertical_flip,
        horizontal_flip,
    ])
    best_psnr = 0
    save_epoch = 0
    best_image = None

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=Gamma)
   
    for i in range(num_epochs):
        model.train()
        aug_image = transform(noisy_image)
        flip_state_1 = vertical_flip.flipped
        flip_state_2 = horizontal_flip.flipped
        mask_image, mask = add_mask(aug_image, mask_ratio, device=device)
        optimizer.zero_grad()
        output = model(mask_image, mask)
        cnt_nonzero = torch.count_nonzero(1-mask)
        loss = torch.sum((output - aug_image)*(output - aug_image)*(1-mask))/cnt_nonzero
        #we only have to divide by cnt_nonzero because the zero values are not counted in the loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        T = 100
        with torch.no_grad():
            if i % 200 == 0:
                if flip_state_1:
                    output= output.flip(1)
                if flip_state_2:
                    output = output.flip(2)
                print('epoch: {}, loss: {}, psnr: {:.4f}, ssim: {}'.format(i, loss.item(), calculate_psnr(image, output), calculate_ssim(image, output)))   

            if (i+1)%1000 == 0:
                avg = torch.zeros_like(image)
                for j in range(T):
                    aug_image = noisy_image#transform(noisy_image)
                    flip_state_1 = vertical_flip.flipped*0
                    flip_state_2 = horizontal_flip.flipped*0
                    mask_image, mask = add_mask(aug_image, mask_ratio, device=device)
                    output_pred = model(mask_image, mask)
                    if flip_state_1:
                        output_pred= output_pred.flip(1)
                    if flip_state_2:
                        output_pred = output_pred.flip(2)
                    avg += output_pred
                    
                avg /= T

                if calculate_psnr(image, avg) > best_psnr:
                    best_psnr = calculate_psnr(image, avg)
                    best_image = avg
                    save_epoch = i
    
    with torch.no_grad():
        save_image(best_image, 'Dropout/{}/{}_best_image_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        save_image(noisy_image, 'Dropout/{}/{}_noisy_image_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        
        print('------------------------------------------------------------')
        print('best image saved at epoch {}, Total samples: {}'.format(save_epoch, T))
        print('best psnr: {:.4f}'.format(best_psnr))
        print('current ssim: {}'.format(calculate_ssim(image, best_image)))
        print('------------------------------------------------------------')

    return best_psnr, calculate_ssim(image, best_image)

def worker(Gamma, step_size, LR, path, file, noise, mask_ratio, num_epochs, device, gray, result_list, exp='exp'):
    if gray:
        channels = 1
    else:
        channels = 3
    model = PartialConvUnet(channels)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_psnr, ssim = train_model(Gamma, step_size, path, file, model, optimizer, noise, mask_ratio, num_epochs, device, gray, exp)
    result_list.append((best_psnr.item() if isinstance(best_psnr, torch.Tensor) else best_psnr, 
                        ssim.item() if isinstance(ssim, torch.Tensor) else ssim))


# Create a global Pool object
pool = None

def signal_handler(sig, frame):
    # Terminate the pool
    if pool is not None:
        pool.terminate()
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)


def main(args):
    dev_list = args.device.split(',')  # multiple device
    devices = ['cuda:{}'.format(dev) for dev in dev_list]
    
    LR = args.lr
    Gamma = args.gamma
    step_size = args.step_size
    num_epochs = args.epoch
    exp = args.exp

    path = 'trainset'
    file_name = os.listdir(path)
    
    noise_lvl = [25, 50, 75, 100]
    mask_ratio = 0.3
    gray = False

    print('------------------------------------------------------------')
    print('learning rate: {}'.format(LR))
    print('gamma:{} {}'.format(step_size, Gamma))
    print('Experiment: {}'.format(exp))
    print('devices: {}'.format(devices))
    print('gray: {}'.format(gray))
    print('mask ratio: {}'.format(mask_ratio))
    print('num epochs: {}'.format(num_epochs))
    print('files: {}'.format(file_name))
    print('------------------------------------------------------------')

    mp.set_start_method('spawn')
    for noise in noise_lvl:
        print('         noise level: {}'.format(noise))
        print('------------------------------------------------------------')

        manager = mp.Manager()
        result_list = manager.list()
        processes = []

        for i, file in enumerate(file_name):
            device = devices[i % len(devices)]  # distribute files to devices
            p = mp.Process(target=worker, args=(Gamma, step_size, LR, path, file, noise, mask_ratio, num_epochs, device, gray, result_list, exp))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = list(result_list)
        average_psnr = sum([psnr for psnr, ssim in results]) / len(results)
        average_ssim = sum([ssim for psnr, ssim in results]) / len(results)

        print('Average PSNR:', average_psnr)
        print('Average SSIM:', average_ssim)
        print('------------------------------------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameterize learning rate, gamma, and step size.')
    parser.add_argument('--device', type=str, default='0,1', help='Comma separated list of devices, e.g., "0,1,2"')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma')
    parser.add_argument('--step_size', type=int, default=2000, help='Step size')
    parser.add_argument('--epoch', type=int, default=200, help='num_epochs')
    parser.add_argument('--exp', type=str, default='exp_multi_device', help='folder for experiment')

    args = parser.parse_args()
    main(args)
