import torch
import torch.nn as nn
import torch.nn.functional as F
 
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.window_size = kernel_size * kernel_size

        nn.init.constant_(self.mask_conv.weight, 1.0)
        #initial to be xavier_uniform for input_conv
        nn.init.xavier_uniform_(self.input_conv.weight)

        for param in self.mask_conv.parameters():
            param.requires_grad = False
        
    def forward(self, input, mask):
        input = F.pad(input, (1,1,1,1), mode='reflect')
        mask = F.pad(mask, (1,1,1,1), mode='constant', value=0)#zero_padding

        output = self.input_conv(input * mask)

        with torch.no_grad():
            new_mask = self.mask_conv(mask)

        new_mask = torch.clip(new_mask, 0, 1)
        mask_ratio = self.window_size / (new_mask + 1e-8)
        mask_ratio = mask_ratio * new_mask
        # Never miss that the bias would be affected by the mask_ratio!
        if self.input_conv.bias is not None:
          bias_ = self.input_conv.bias.view(1,-1,1,1)
          output = (output-bias_) * mask_ratio + bias_
        else:
          output = output * mask_ratio

        return output, new_mask
'''
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.window_size = kernel_size * kernel_size

        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False
        
    def forward(self, input, mask):

        input = F.pad(input, (1,1,1,1), mode='reflect')
        mask = F.pad(mask, (1,1,1,1), mode='constant', value=1)

        output = self.input_conv(input * mask)

        with torch.no_grad():
            new_mask = self.mask_conv(mask)

        new_mask = torch.clip(new_mask, 0, 1)
        mask_ratio = self.window_size / (new_mask + 1e-8)
        mask_ratio = mask_ratio * new_mask
        output = output * mask_ratio
        return output, new_mask
'''

class PartialEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Pconv = PartialConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, mask):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')  # tf.pad with "SYMMETRIC" corresponds to 'reflect' mode in PyTorch
        mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=1)  # tf.pad with "CONSTANT" and constant_values=1
        x, mask = self.Pconv(x, mask)
        #print(mask.shape)
        x = self.leaky_relu(x)
        x = self.maxpool(x)
        mask = self.maxpool(mask)
        return x, mask
    
def custom_weights_init(layer, gain=2):
    shape = layer.weight.shape
    fan_in = torch.prod(torch.tensor(shape[:-1])).item()
    std = torch.sqrt(torch.tensor(gain / fan_in))
    nn.init.normal_(layer.weight, 0.0, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class PartialDecoder(nn.Module):
    def __init__(self, in_, out_1, out_2, p = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, out_1, 3, 1, 1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(out_1, out_2, 3, 1, 1, padding_mode='reflect')
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

        self.conv1.apply(custom_weights_init)
        self.conv2.apply(custom_weights_init)

        '''
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)
        '''

    def concat(self, x, y):
        bs1, c1, h1, w1 = x.size()
        bs2, c2, h2, w2 = y.size()

        min_h = min(h1, h2)
        min_w = min(w1, w2)

        x_cropped = x[:, :, :min_h, :min_w]
        y_cropped = y[:, :, :min_h, :min_w]

        result = torch.cat((x_cropped, y_cropped), dim=1)

        return result


    def forward(self, x, skip):
        x = self.upsample(x)
        #print(x.shape)
        #print(skip.shape)
        x = self.concat(x, skip)
        #x = torch.cat([x,skip],1)
        #print(x.shape)

        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        return x  

class PartialConvUnet(nn.Module):
    def __init__(self, channel = 3, p=0.3):
        super().__init__()
        self.Pconv0 = PartialConv(channel, 48)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.encoder1 = PartialEncoder(48, 48)
        self.encoder2 = PartialEncoder(48, 48)
        self.encoder3 = PartialEncoder(48, 48)
        self.encoder4 = PartialEncoder(48, 48)
        self.encoder5 = PartialEncoder(48, 48)

        self.bottleneck = PartialConv(48, 48)

        self.decoder1 = PartialDecoder(48+48, 96, 96)
        self.decoder2 = PartialDecoder(48+96, 96, 96)
        self.decoder3 = PartialDecoder(48+96, 96, 96)
        self.decoder4 = PartialDecoder(48+96, 96, 96)
        self.decoder5 = PartialDecoder(channel+96, 64, 32)

        #I forgot this dropout layer
        self.final_conv = nn.Sequential(nn.Dropout(p),
            nn.Conv2d(32, channel, 3, 1, 1, padding_mode='reflect'), 
            nn.Sigmoid())
        
        self.final_conv[1].apply(lambda layer: custom_weights_init(layer, gain=1.0))

        '''
        nn.init.kaiming_normal_(self.final_conv[0].weight, mode='fan_in', nonlinearity='leaky_relu')    
        if self.final_conv[0].bias is not None:
            nn.init.constant_(self.final_conv[0].bias, 0)
        '''
    

    def forward(self, x, mask):
        skips = [x]

        n = x
        n, mask = self.Pconv0(n, mask)
        n = self.leaky_relu(n)
        n, mask = self.encoder1(n, mask)
        skips.append(n)
        n, mask = self.encoder2(n, mask)
        skips.append(n)
        n, mask = self.encoder3(n, mask)
        skips.append(n)
        n, mask = self.encoder4(n, mask)
        skips.append(n)
        n, mask = self.encoder5(n, mask)

        n, mask = self.bottleneck(n, mask)
        n = self.leaky_relu(n)

        n = self.decoder1(n, skips.pop())
        n = self.decoder2(n, skips.pop())
        n = self.decoder3(n, skips.pop())
        n = self.decoder4(n, skips.pop())
        n = self.decoder5(n, skips.pop())

        n = self.final_conv(n)

        return n


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, num_of_features=64, dropout_prob=0.3):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=num_of_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_prob))
        
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_of_features))
            layers.append(nn.ReLU(inplace=True))
            if _ > 8:
                layers.append(nn.Dropout(p=dropout_prob))
        
        layers.append(nn.Conv2d(in_channels=num_of_features, out_channels=channels, kernel_size=3, padding=1, bias=False))
        cnt = 0
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                if cnt <7:
                    nn.init.xavier_normal_(layer.weight)
                    
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                cnt+=1

        self.dncnn = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.sigmoid(self.dncnn(x))
        return out
