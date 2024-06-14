import torch
import torch.nn as nn
import torch.nn.functional as F
#https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py#
'''
class BayesianConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BayesianConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_sigma = torch.log1p(torch.exp(self.rho))
        self.weight_mu = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.uniform_(self.rho, a=-5, b=-4)
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        

        self.weight_epsilon = torch.randn_like(self.weight_mu)
        self.weight = self.weight_mu + self.weight_sigma * self.weight_epsilon

    def forward(self, x):
        if self.training:
            self.weight_epsilon = torch.randn_like(self.weight_mu)
            self.weight_sigma = torch.log1p(torch.exp(self.rho))
            self.weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
'''

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, padding_mode='zeros'):
        super().__init__()
        if padding_mode not in ['zeros', 'reflect', 'replicate', 'circular']:
            raise ValueError('padding_mode must be one of "zeros", "reflect", "replicate", or "circular"')
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d((2, 2))
        

    def forward(self, inputs):
        p = inputs.clone()
        x = self.conv(inputs)
        x = self.leaky_relu(x)
        x = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c_1, out_c_2, rate = 0, padding_mode='zeros'):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_c, out_c_1, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_c_1, out_c_2, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.rate = rate
        if self.rate < 0 or self.rate > 1:
            raise ValueError('rate must be between 0 and 1')
        elif self.rate == 0:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
        else: 
            self.dropout1 = nn.Dropout(rate)
            self.dropout2 = nn.Dropout(rate)
        

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat((x, skip), dim=1)
        
        x = self.dropout1(x)*(1-self.rate)#due to dropout, tf 1.x different from tf 2.x by keep_prob and *1/(keep_prob).
        x = self.leaky_relu(self.conv1(x))
        x = self.dropout2(x)*(1-self.rate)
        x = self.leaky_relu(self.conv2(x))
        '''
        print(x.shape)
        print(skip.shape)
        '''

        return x

class Unet(nn.Module):
    def __init__(self, channels, padding_mode='zeros'):
        super(Unet, self).__init__()

        self.start = nn.Conv2d(channels, 48, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.encoder1 = encoder_block(48, 48)
        self.encoder2 = encoder_block(48, 48)
        self.encoder3 = encoder_block(48, 48)
        self.encoder4 = encoder_block(48, 48)
        self.encoder5 = encoder_block(48, 48)

        self.bottleneck = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        self.decoder1 = decoder_block(48+48, 96, 96)
        self.decoder2 = decoder_block(96+48, 96, 96)
        self.decoder3 = decoder_block(96+48, 96, 96)
        self.decoder4 = decoder_block(96+48, 96, 96)
        self.decoder5 = decoder_block(96+channels, 64, 32)

        self.final_conv = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        p1 = x.clone()

        x = self.leaky_relu(self.start(x))

        x, _ = self.encoder1(x)
        x, p2 = self.encoder2(x)
        x, p3 = self.encoder3(x)
        x, p4 = self.encoder4(x)
        x, p5 = self.encoder5(x)

        x = self.leaky_relu(self.bottleneck(x))

        x = self.decoder1(x, p5)
        x = self.decoder2(x, p4)
        x = self.decoder3(x, p3)
        x = self.decoder4(x, p2)
        x = self.decoder5(x, p1)

        x = self.leaky_relu(self.final_conv(x))
        x = self.sigmoid(x)

        return x
        

class Dropout_Unet(nn.Module):
    def __init__(self, channels, rate = 0.3, padding_mode='zeros'):
        super(Dropout_Unet, self).__init__()

        self.start = nn.Conv2d(channels, 48, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.encoder1 = encoder_block(48, 48)
        self.encoder2 = encoder_block(48, 48)
        self.encoder3 = encoder_block(48, 48)
        self.encoder4 = encoder_block(48, 48)
        self.encoder5 = encoder_block(48, 48)

        self.bottleneck = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        self.decoder1 = decoder_block(48+48, 96, 96, rate=rate)
        self.decoder2 = decoder_block(96+48, 96, 96, rate=rate)
        self.decoder3 = decoder_block(96+48, 96, 96, rate=rate)
        self.decoder4 = decoder_block(96+48, 96, 96, rate=rate)
        self.decoder5 = decoder_block(96+channels, 64, 32, rate=rate)
        self.dropout = nn.Dropout(rate)

        self.final_conv = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)


        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        p1 = x.clone()

        x = self.leaky_relu(self.start(x))

        x, _ = self.encoder1(x)
        x, p2 = self.encoder2(x)
        x, p3 = self.encoder3(x)
        x, p4 = self.encoder4(x)
        x, p5 = self.encoder5(x)

        x = self.leaky_relu(self.bottleneck(x))

        x = self.decoder1(x, p5)
        x = self.decoder2(x, p4)
        x = self.decoder3(x, p3)
        x = self.decoder4(x, p2)
        x = self.decoder5(x, p1)

        x = self.dropout(x)

        x = self.leaky_relu(self.final_conv(x))
        x = self.sigmoid(x)

        return x

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.energy = kernel_size * kernel_size

        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):

        output = self.input_conv(input * mask)


        with torch.no_grad():
            output_mask = self.mask_conv(mask)


        output_mask = torch.where(output_mask == 0, torch.ones_like(output_mask) * 1e-8, output_mask)


        output = output / output_mask * self.energy


        new_mask = torch.where(output_mask == 0, torch.zeros_like(output_mask), torch.ones_like(output_mask))

        return output, new_mask