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

import sys

from MLE.utils import RandomVerticalFlipWithState, RandomHorizontalFlipWithState, read_image, calculate_psnr, calculate_ssim, add_noise, add_mask, average_masked_image
import torch
import torch.nn.functional as F
from MLE.model import Dropout_Unet
import signal
import multiprocessing

def train_model(path, file_name, model, optimizer, criterion, noise_lvl, mask_ratio, num_epochs, device='cuda:1', gray=False, average=False):
    
    sys.stdout = open(f'Dropout/{noise_lvl}_{file_name[:-4]}_{average}.txt', 'w', buffering=1)
    print('------------------------------------------------------------')
    print('noise level: {}'.format(noise_lvl))
    print('file name: {}'.format(file_name))
    print('mask ratio: {}'.format(mask_ratio))
    print('num epochs: {}'.format(num_epochs))
    print('gray: {}'.format(gray))
    print('average: {}'.format(average))
    print('------------------------------------------------------------')
    file_path = os.path.join(path, file_name)
    image = read_image(file_path, device=device, gray=gray)
    noisy_image = add_noise(image, noise_lvl)
    vertical_flip = RandomVerticalFlipWithState(1/3)
    horizontal_flip = RandomHorizontalFlipWithState(1/3)
    transform = transforms.Compose([
        vertical_flip,
        horizontal_flip,
    ])
    best_psnr = 0
    best_image = None
   
    for i in range(num_epochs):
        model.train()
        aug_image = transform(noisy_image)
        flip_state_1 = vertical_flip.flipped
        flip_state_2 = horizontal_flip.flipped
        mask_image, mask = add_mask(aug_image, mask_ratio)
        if average:
            mask_image = average_masked_image(mask_image, mask)
        optimizer.zero_grad()
        output = model(mask_image)
        #cnt_nonzero = torch.count_nonzero(1-mask)
        loss = criterion(output*(1-mask), aug_image*(1-mask))#/cnt_nonzero*noisy_image.size(2)*noisy_image.size(3)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if i % 200 == 0:
                if flip_state_1:
                    output= output.flip(1)
                if flip_state_2:
                    output = output.flip(2)
                print('epoch: {}, loss: {}, psnr: {:.4f}, ssim: {}'.format(i, loss.item(), calculate_psnr(image, output), calculate_ssim(image, output)))   

            
            T = 100
            avg = torch.zeros_like(image)
            for j in range(T):
                aug_image = transform(noisy_image)
                flip_state_1 = vertical_flip.flipped
                flip_state_2 = horizontal_flip.flipped
                mask_image, mask = add_mask(aug_image, mask_ratio)
                if average:
                    mask_image = average_masked_image(mask_image, mask)
                output_pred = model(mask_image)
                if flip_state_1:
                    output_pred= output_pred.flip(1)
                if flip_state_2:
                    output_pred = output_pred.flip(2)
                avg += output_pred/T

            if calculate_psnr(image, sum) > best_psnr:
                best_psnr = calculate_psnr(image, sum)
                best_image = avg
    
    with torch.no_grad():
        save_image(best_image, 'Dropout/{}_best_image_{}_{}.png'.format(noise_lvl,file_name[:-4],average))
        save_image(noisy_image, 'Dropout/{}_noisy_image_{}_{}.png'.format(noise_lvl,file_name[:-4],average))
        print('------------------------------------------------------------')
        print('best image saved, Total samples: {}'.format(T))
        print('best psnr: {:.4f}'.format(best_psnr))
        print('current ssim: {}'.format(calculate_ssim(image, best_image)))
        print('------------------------------------------------------------')

    return best_psnr, calculate_ssim(image, best_image)

def worker(path, file, noise, mask_ratio, num_epochs, device, gray, average):
    model = Dropout_Unet(channels, rate = 0.3)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    best_psnr, ssim = train_model(path, file, model, optimizer, criterion, noise, mask_ratio, num_epochs, device, gray, average)
    return best_psnr, ssim

# Create a global Pool object
pool = None

def signal_handler(sig, frame):
    # Terminate the pool
    if pool is not None:
        pool.terminate()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    path = '../Dataset/trainset'
    file_name = os.listdir('../Dataset/trainset')
    channels = 3
    num_epochs = 150000
    noise_lvl = [25, 50]
    mask_ratio = 0.3
    gray = False
    average = True
    print('------------------------------------------------------------')
    print('device: {}'.format(device))
    print('average: {}'.format(average))
    print('gray: {}'.format(gray))
    print('mask ratio: {}'.format(mask_ratio))
    print('num epochs: {}'.format(num_epochs))
    print('files: {}'.format(file_name))
    print('------------------------------------------------------------')
    for noise in noise_lvl:
        print('         noise level: {}'.format(noise))
        print('------------------------------------------------------------')
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(4) as pool:
            results = pool.starmap(worker, [(path, file, noise, mask_ratio, num_epochs, device, gray, average) for file in file_name])
        average_psnr = sum([psnr for psnr, ssim in results])/len(results)
        average_ssim = sum([ssim for psnr, ssim in results])/len(results)
        print('Average PSNR:', average_psnr)
        print('Average SSIM:', average_ssim)
        print('------------------------------------------------------------')

