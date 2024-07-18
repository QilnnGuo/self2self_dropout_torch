import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import signal
import multiprocessing
import argparse

from torchvision import transforms
from torchvision.utils import save_image
from model.utils import add_mask, read_image, RandomVerticalFlipWithState, RandomHorizontalFlipWithState, add_noise, calculate_psnr, calculate_ssim
from model.model import DnCNN

def train_model(Gamma, step_size, path, file_name, model, optimizer, criterion, noise_lvl, mask_ratio, num_epochs, device='cuda:1', gray=False, exp = 'exp'):
    os.makedirs(f'Dropout/{exp}', exist_ok=True)
    sys.stdout = open(f'Dropout/{exp}/{noise_lvl}_{file_name[:-4]}.txt', 'w', buffering=1)
    psnr = []
    ssim = []
    Loss = []
    one_epoch_psnr = []
    one_epoch_ssim = []
    predicted_psnr = []
    predicted_ssim = []
    
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
    #of no use
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=Gamma)
   
    for i in range(num_epochs):
        model.train()
        aug_image = transform(noisy_image)
        flip_state_1 = vertical_flip.flipped
        flip_state_2 = horizontal_flip.flipped
        mask_image, mask = add_mask(aug_image, mask_ratio, device=device)
        optimizer.zero_grad()
        output = model(mask_image)
        cnt_nonzero = torch.count_nonzero(1-mask)
        loss = torch.sum((output - aug_image)**2*(1-mask))/cnt_nonzero
        save_image(mask_image, 'Dropout/{}/{}_mask_image_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        save_image(aug_image, 'Dropout/{}/{}_aug_image_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        save_image(output, 'Dropout/{}/{}_output_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        T = 100
        with torch.no_grad():

            if flip_state_1:
                output= output.flip(2)
            if flip_state_2:
                output = output.flip(3)
            psnr_val = calculate_psnr(image, output)
            ssim_val = calculate_ssim(image, output)
            psnr.append(psnr_val.item())
            ssim.append(ssim_val.item())
            Loss.append(loss.item())

            if i % 200 == 0:
                print('epoch: {}, loss: {}, psnr: {:.4f}, ssim: {}'.format(i, loss.item(), psnr_val, ssim_val))

            if (i+1)%500 == 0:
                avg = torch.zeros_like(image)
                for j in range(T):
                    aug_image = noisy_image#transform(noisy_image)
                    flip_state_1 = vertical_flip.flipped*0
                    flip_state_2 = horizontal_flip.flipped*0
                    mask_image, mask = add_mask(aug_image, mask_ratio, device=device)
                    output_pred = model(mask_image)
                    if flip_state_1:
                        output_pred= output_pred.flip(2)
                    if flip_state_2:
                        output_pred = output_pred.flip(3)
                    
                    if i == 49999:
                        pred_psnr = calculate_psnr(image, output_pred)
                        pred_ssim = calculate_ssim(image, output_pred)
                        one_epoch_psnr.append(pred_psnr.item())
                        one_epoch_ssim.append(pred_ssim.item())

                    avg += output_pred
                avg /= T

                avg_psnr = calculate_psnr(image, avg)
                avg_ssim = calculate_ssim(image, avg)
                predicted_psnr.append(avg_psnr.item())
                predicted_ssim.append(avg_ssim.item())

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_image = avg
                    save_epoch = i
                    print('saved at epoch: {}, best_psnr: {:.4f}, ssim: {}'.format(i,best_psnr, avg_ssim))
    
    with torch.no_grad():
        save_image(best_image, 'Dropout/{}/{}_best_image_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        save_image(noisy_image, 'Dropout/{}/{}_noisy_image_{}.png'.format(exp,noise_lvl,file_name[:-4]))
        
        print('------------------------------------------------------------')
        print('best image saved at epoch {}, Total samples: {}'.format(save_epoch, T))
        print('best psnr: {:.4f}'.format(best_psnr))
        print('current ssim: {}'.format(calculate_ssim(image, best_image)))
        print('------------------------------------------------------------')

    #plot psnr, ssim, predicted_psnr, predicted_ssim using plt.subplots
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(psnr)
    plt.title('PSNR')
    plt.subplot(2,2,2)
    plt.plot(ssim)
    plt.title('SSIM')
    plt.subplot(2,2,3)
    plt.plot(predicted_psnr)
    #draw a vertical line at save_epoch
    plt.axvline(x=(save_epoch+1)/500, color='r', linestyle='--')
    plt.title('Predicted PSNR')
    plt.subplot(2,2,4)
    plt.plot(predicted_ssim)
    #draw a vertical line at save_epoch
    plt.axvline(x=(save_epoch+1)/500, color='r', linestyle='--')
    plt.title('Predicted SSIM')
    plt.savefig('Dropout/{}/{}_plot_{}.png'.format(exp,noise_lvl,file_name[:-4]))
    plt.close()
    #plot loss by epoch, 49999 epoch psnr, 49999 epoch ssim
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(Loss)
    plt.title('Loss')
    plt.subplot(2,2,2)
    plt.bar(range(len(one_epoch_psnr)), one_epoch_psnr)
    plt.title('49999 Epoch PSNR individual prediction')
    plt.subplot(2,2,3)
    plt.bar(range(len(one_epoch_ssim)), one_epoch_ssim)
    plt.title('49999 Epoch SSIM individual prediction')
    plt.savefig('Dropout/{}/{}_loss_plot_{}.png'.format(exp,noise_lvl,file_name[:-4]))
    plt.close()

    return best_psnr, calculate_ssim(image, best_image)

def worker(Gamma, step_size, LR, path, file, noise, mask_ratio, num_epochs, device, gray, exp='exp'):
    if gray:
        channels = 1
    else:
        channels = 3
    model = DnCNN(channels, num_of_layers=17, num_of_features=64, dropout_prob=0.3)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()#of no use
    best_psnr, ssim = train_model(Gamma, step_size, path, file, model, optimizer, criterion, noise, mask_ratio, num_epochs, device, gray, exp)
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
    parser = argparse.ArgumentParser(description='Parameterize learning rate, gamma, and step size.')
    parser.add_argument('--device', type=int, default=1, help='device')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma')
    parser.add_argument('--step_size', type=int, default=2000, help='Step size')
    parser.add_argument('--epoch', type=int, default=150000, help='num_epochs')
    parser.add_argument('--exp', type=str, default='exp_new', help='folder for experiment')
    parser.add_argument('--lvl', type=str, default='50,75,100', help='noise level')

    args = parser.parse_args()

    dev = args.device
    device = torch.device('cuda:{}'.format(dev) if torch.cuda.is_available() else 'cpu')#cuda:3
    LR = args.lr
    Gamma = args.gamma
    step_size = args.step_size
    num_epochs = args.epoch
    exp = args.exp

    path = '../trainset'
    file_name = os.listdir('../trainset')
    
    noise_lvl = [int(x) for x in args.lvl.split(',')]
    mask_ratio = 0.3
    gray = False

    print('------------------------------------------------------------')
    print('learning rate: {}'.format(LR))
    print('gamma:{} {}'.format(step_size, Gamma))#step_size, Gamma
    print('Experiment: {}'.format(exp))
    print('device: {}'.format(device))
    print('gray: {}'.format(gray))
    print('mask ratio: {}'.format(mask_ratio))
    print('num epochs: {}'.format(num_epochs))
    print('files: {}'.format(file_name))
    print('------------------------------------------------------------')
    multiprocessing.set_start_method('spawn')
    for noise in noise_lvl:
        print('         noise level: {}'.format(noise))
        print('------------------------------------------------------------')
        with multiprocessing.Pool(4) as pool:
            results = pool.starmap(worker, [(Gamma, step_size, LR, path, file, noise, mask_ratio, num_epochs, device, gray, exp) for file in file_name])
        average_psnr = sum([psnr for psnr, ssim in results])/len(results)
        average_ssim = sum([ssim for psnr, ssim in results])/len(results)
        print('Average PSNR:', average_psnr)
        print('Average SSIM:', average_ssim)
        print('------------------------------------------------------------')

