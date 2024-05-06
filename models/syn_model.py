

import torch.nn as nn
import torch.nn.functional as F

class PreConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_type, act_type, norm_type, up_block, ksize=3):
        super(PreConvBlock, self).__init__()
        self.up_type = up_type
        self.act_type = act_type
        self.norm_type = norm_type

        if self.norm_type is not None:
            if self.norm_type == "bn":
                self.bn = nn.BatchNorm2d(in_channels)
            elif self.norm_type == "in":
                self.bn = nn.InstanceNorm2d(in_channels)
            else:
                raise NotImplementedError(self.norm_type)
        
        # ACT_TYPE = {0: None, 1: "relu", 2: "sig", 3: "tanh"}
        if self.act_type is not None:
            if self.act_type == "relu":
                self.act = nn.ReLU()
            elif self.act_type == 'sig':
                self.act = nn.Sigmoid()
            elif self.act_type == 'elu':
                self.act = nn.Elu()
            elif self.act_type == 'tanh':
                self.act = nn.Tanh()
            else:
                raise NotImplementedError(self.act_type)

        self.up_block = up_block
        if self.up_type == "deconv":
            self.deconv = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize // 2)


    def forward(self, x):
        # norm
        if self.norm_type is not None:
            h = self.bn(x)
        else:
            h = x

        # activation
        if self.act_type is not None:
            h = self.act(h)

        # whether this is a upsample block
        if self.up_block:
            if self.up_type == "deconv":
                h = self.deconv(h)
            else:
                h = F.interpolate(h, scale_factor=2, mode=self.up_type)

        # conv
        out = self.conv(h)
        return out


class PostConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_type, act_type, norm_type, up_block, ksize=3):
        super(PostConvBlock, self).__init__()
        self.up_type = up_type
        self.act_type = act_type
        self.norm_type = norm_type

        self.up_block = up_block
        if self.up_type == "deconv":
            self.deconv = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize // 2)
        
        if self.norm_type is not None:
            if self.norm_type == "bn":
                self.bn = nn.BatchNorm2d(out_channels)
            elif self.norm_type == "in":
                self.bn = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(self.norm_type)

        if self.act_type is not None:
            if self.act_type == "relu":
                self.act = nn.ReLU()
            elif self.act_type == 'sig':
                self.act = nn.Sigmoid()
            elif self.act_type == 'elu':
                self.act = nn.Elu()
            elif self.act_type == 'tanh':
                self.act = nn.Tanh()
            else:
                raise NotImplementedError(self.act_type)


    def forward(self, x):
        # whether this is a upsample block
        if self.up_block:
            if self.up_type == "deconv":
                h = self.deconv(x)
            else:
                h = F.interpolate(x, scale_factor=2, mode=self.up_type)
        else:
            h = x

        # conv
        h = self.conv(h)

        # norm
        if self.norm_type is not None:
            h = self.bn(h)

        # activation
        if self.act_type is not None:
            out = self.act(h)
        else:
            out = h

        return out


class GBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(GBlock, self).__init__()

        if config.conv_type == 'post':

            if config.layer_num == 2:
                self.conv1 = PostConvBlock(
                    in_channels, in_channels, \
                    config.up_type, config.act_type, config.norm_type, \
                    up_block=True, ksize=config.kernel_size
                )
                self.conv2 = PostConvBlock(
                    in_channels, out_channels, \
                    config.up_type, config.act_type, config.norm_type, \
                    up_block=False, ksize=config.kernel_size
                )
            else:
                self.conv1 = PostConvBlock(
                    in_channels, out_channels, \
                    config.up_type, config.act_type, config.norm_type, \
                    up_block=True, ksize=config.kernel_size
                )

        elif config.conv_type == 'pre':
            if config.layer_num == 2:
                self.conv1 = PreConvBlock(
                    in_channels, in_channels, \
                    config.up_type, config.act_type, config.norm_type, \
                    up_block=True, ksize=config.kernel_size
                )
                self.conv2 = PreConvBlock(
                    in_channels, out_channels, \
                    config.up_type, config.act_type, config.norm_type, \
                    up_block=False, ksize=config.kernel_size
                )
            else:
                self.conv1 = PreConvBlock(
                    in_channels, out_channels, \
                    config.up_type, config.act_type, config.norm_type, \
                    up_block=True, ksize=config.kernel_size
                )

        self.conv_type = config.conv_type
        self.layer_num = config.layer_num
        self.config = config


    def forward(self, x):

        h = self.conv1(x)
        if self.layer_num == 2:
            h = self.conv2(h)

        return h

class DBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(DBlock, self).__init__()
    
        self.c1 = nn.Conv2d(in_channels, out_channels, config.kernel_size, 1, padding=config.kernel_size//2)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(True)
        self.c2 = nn.Conv2d(out_channels, out_channels, config.kernel_size, 1, padding=config.kernel_size//2)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        h = F.avg_pool2d(x, 2)
        h = self.c1(h)
        h = self.b1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = self.b2(h)

        return h


class SynModel(nn.Module):
    def __init__(self, config):
        super(SynModel, self).__init__()

        if config.block_num == 1:
            self.Dblock1 = DBlock(config, 3, config.in_channels)
            self.Gblock1 = GBlock(config, config.in_channels, 3)
            
        elif config.block_num == 2:
            self.Dblock1 = DBlock(config, 3, config.in_channels)
            self.Dblock2 = DBlock(config, config.in_channels, 2 * config.in_channels)

            self.Gblock1 = GBlock(config, 2 * config.in_channels, config.in_channels)
            self.Gblock2 = GBlock(config, config.in_channels, 3)

        self.config = config

    def forward(self, x):

        if self.config.block_num == 1:

            h = self.Dblock1(x)
            output = self.Gblock1(h)

        elif self.config.block_num == 2:

            x = self.Dblock1(x)
            h = self.Dblock2(x)

            x = self.Gblock1(h)
            output = self.Gblock2(x)

        return output



