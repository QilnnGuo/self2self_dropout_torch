import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F
from math import exp
def add_noise(image, noise_level, device='cuda:1'):#image float tensor
    noise =  torch.randn(image.size()) * noise_level/255
    noise = noise.to(device)
    noisy_image = image + noise
    #noisy_image = torch.clip(noisy_image, 0, 1)
    return noisy_image.to(device)

def add_mask(image, mask_ratio, device='cuda:1'):
    mask = torch.rand(image.size())
    mask[mask < mask_ratio] = 0
    mask[mask >= mask_ratio] = 1
    mask = mask.to(device)
    masked_image = image * mask
    return masked_image, mask

def read_image(image_path, transform=transforms.ToTensor(), device='cuda:1', gray=False):
    if gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).float()
    return image.to(device)


def average_masked_image(mask_image, mask, device='cuda:1'):
    padded_image = F.pad(mask_image, (1, 1, 1, 1), mode='constant', value=0)
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)

    kernel = torch.ones((mask_image.shape[1], mask_image.shape[1], 3, 3), device=device)
    neighbor_sum = F.conv2d(padded_image, kernel, padding=0)

    neighbor_count = F.conv2d(padded_mask, kernel, padding=0)

    filled_image = mask_image.clone()
    mask_indices = (mask == 0)

    # avoid divide by zero
    neighbor_count[neighbor_count == 0] = 1  

    filled_image[mask_indices] = neighbor_sum[mask_indices] / neighbor_count[mask_indices]
    return filled_image

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class RandomVerticalFlipWithState():
    def __init__(self, probability):
        self.probability = probability
        self.flipped = False

    def __call__(self, img):
        if torch.rand(1) < self.probability:
            self.flipped = True
            return img.flip(1)
        self.flipped = False
        return img

class RandomHorizontalFlipWithState():
    def __init__(self, probability):
        self.probability = probability
        self.flipped = False

    def __call__(self, img):
        if torch.rand(1) < self.probability:
            self.flipped = True
            return img.flip(2)
        self.flipped = False
        return img