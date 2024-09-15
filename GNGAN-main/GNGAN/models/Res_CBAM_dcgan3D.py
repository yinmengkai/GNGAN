import torch
import torch.nn as nn


# 残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        out = self.sigmoid(x)
        return out


# CBAM注意力机制
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out


# 生成器
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.network = nn.Sequential(
            nn.ConvTranspose3d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(opt.ngf * 8),

            nn.ConvTranspose3d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(opt.ngf * 4),
            nn.ReLU(True),

            ResidualBlock(opt.ngf * 4, opt.ngf * 4),
            CBAM(opt.ngf * 4),

            nn.ConvTranspose3d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(opt.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose3d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(opt.ngf),
            nn.ReLU(True),

            ResidualBlock(opt.ngf, opt.ngf),
            CBAM(opt.ngf),

            nn.ConvTranspose3d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        output = self.network(inputs)
        return output


# 判别器
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.network = nn.Sequential(
            nn.Conv3d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            ResidualBlock(opt.ndf, opt.ndf),
            CBAM(opt.ndf),

            nn.Conv3d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(opt.ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(opt.ndf * 4),
            nn.LeakyReLU(0.2, True),

            ResidualBlock(opt.ndf * 4, opt.ndf * 4),
            CBAM(opt.ndf * 4),

            nn.Conv3d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(opt.ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, inputs):
        output = self.network(inputs)
        output = output.view(-1, 1).squeeze(1)
        return output
