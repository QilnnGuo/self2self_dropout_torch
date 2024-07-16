import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import add_mask, read_image, RandomVerticalFlipWithState, RandomHorizontalFlipWithState, add_noise, calculate_psnr, calculate_ssim
from model.model import PartialConvUnet
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import sys

def train(image, path, noise_lvl, num_epochs, device, model, optimizer, scheduler, average=1):
    sys.stdout = open(f'{noise_lvl}_{device}.txt', 'w', buffering=1)
    print('noise level: {}'.format(noise_lvl))
    noisy_image = add_noise(image, noise_lvl, device=device)
    vertical_flip = RandomVerticalFlipWithState(1/3)
    horizontal_flip = RandomHorizontalFlipWithState(1/3)
    transform = transforms.Compose([
        vertical_flip,
        horizontal_flip,
        ])
    best_psnr = 0
    best_image = None
    save_epoch = 0
    T = 100
    mask_ratio = 0.3
    criterion = nn.MSELoss()
    for i in range(num_epochs):
        model.train()
        loss = 0
        aug_image = transform(noisy_image)
        flip_state_1 = vertical_flip.flipped
        flip_state_2 = horizontal_flip.flipped
        mask_image, mask = add_mask(aug_image, mask_ratio, device=device)
        optimizer.zero_grad()
        output = model(mask_image, mask)
        if average:
            cnt_nonzero = torch.count_nonzero(1-mask)
        else:
            cnt_nonzero = 1

        loss = criterion(output*(1-mask), aug_image*(1-mask))/cnt_nonzero
            
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if (i+1) % 100 == 0:
                if flip_state_1:
                    output= output.flip(1)
                if flip_state_2:
                    output = output.flip(2)
                print('epoch: {}, loss: {}, psnr: {:.4f}, ssim: {}'.format(i, loss.item(), calculate_psnr(image, output), calculate_ssim(image, output)))   
            
            if i%500 == 0:
                avg = torch.zeros_like(image)
                for j in range(T):
                    aug_image = noisy_image#transform(noisy_image)
                    flip_state_1 = vertical_flip.flipped*0
                    flip_state_2 = horizontal_flip.flipped*0
                    mask_image, mask = add_mask(aug_image, mask_ratio, device=device)
                    output_pred = model(mask_image,mask)
                    if flip_state_1:
                        output_pred= output_pred.flip(1)
                    if flip_state_2:
                        output_pred = output_pred.flip(2)
                    avg += output_pred/T

                if calculate_psnr(image, avg) > best_psnr:
                    best_psnr = calculate_psnr(image, avg)
                    best_image = avg
                    save_epoch = i
            
    with torch.no_grad():
        print('------------------------------------------------------------')
        print('best image of {} saved at epoch {}, Total samples: {}'.format(path, save_epoch, T))
        print('best psnr: {:.4f}'.format(best_psnr))
        print('current ssim: {}'.format(calculate_ssim(image, best_image)))
        print('------------------------------------------------------------')
        save_image(best_image, '{}_best_{}.png'.format(noise_lvl, path[9:-4]))
        save_image(noisy_image, '{}_noisy_{}.png'.format(noise_lvl, path[9:-4]))
    return

#for i in noise_lvl:
#    train(i)
if __name__ == '__main__':

    path = 'trainset/kodim01.png'#'trainset/Peppers.png'
    print('------------------------------------------------------------')


    parser = argparse.ArgumentParser(description='Parameterize learning rate, gamma, and step size.')
    parser.add_argument('--path', type=str, default=path, help='path')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma')
    parser.add_argument('--step_size', type=int, default=3000, help='Step size')
    parser.add_argument('--epoch', type=int, default=450000, help='num_epochs')
    parser.add_argument('--M', type=int, default=10, help='samples')
    parser.add_argument('--average', type=int, default=0, help='loss average')

    args = parser.parse_args()

    path = args.path

    num_epochs = args.epoch
    print('file name: {}'.format(path))

    dev = args.device
    device = torch.device('cuda:{}'.format(dev) if torch.cuda.is_available() else 'cpu')#cuda:3
    average = args.average
    LR = args.lr
    GAMMA = args.gamma
    step_size = args.step_size
    M = args.M

    model = PartialConvUnet()
    model.to(device)
    loss_fn = nn.MSELoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=GAMMA)
    image = read_image(path, device=device)
    noise_lvl = [50,75,100]
    print('------------------------------------------------------------')
    print('device: {}'.format(device))
    print('num_epoch={}, learning rate = {}, gamma = {}, step_size = {}, average = {}'.format(args.epoch, LR, GAMMA, step_size, average))
    print('------------------------------------------------------------')
    dev = args.device
    device = torch.device('cuda:{}'.format(dev) if torch.cuda.is_available() else 'cpu')#cuda:3
    average = args.average
    LR = args.lr
    GAMMA = args.gamma
    step_size = args.step_size
    M = args.M
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(3) as pool:
        pool.starmap(train, [(image, path, i, num_epochs, device, model, optimizer, scheduler, average) for i in noise_lvl])
