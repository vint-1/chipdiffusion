import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks

class UNet(nn.Module): # UNet class used for diffusion models

    def __init__(self, in_channels, out_channels, image_shape, cnn_depths, layers_per_block, filter_size, pooling_factor, cond_dim, level_block="conv", device="cpu", **kwargs):
        # length of CNN_depths determines how many levels u-net has
        super().__init__()
        self._down_conv_blocks = []
        self._up_conv_blocks = []
        self._transpose_conv_layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_shape = image_shape
        self.cnn_depths = cnn_depths
        self.filter_size = filter_size
        self.pooling_factor = pooling_factor
        self.level_block = level_block
        self.encoding_dim = cond_dim

        # create downward branch
        level_in_shape = (in_channels, *image_shape)
        for i, cnn_depth in enumerate(cnn_depths):
            if self.level_block == "conv":
                level_net = networks.ConvNet(
                    level_in_shape[0]+cond_dim if i==(len(cnn_depths)-1) else level_in_shape[0], 
                    level_in_shape[1:], 
                    [filter_size] * layers_per_block,
                    [1] * layers_per_block,
                    [cnn_depth] * layers_per_block,
                    padding = "same",
                    **kwargs
                    )
                level_out_shape = level_net.out_shape
            elif self.level_block == "res":
                level_blocks = [networks.ResBlock(
                    in_channels=(level_in_shape[0]+cond_dim if i==(len(cnn_depths)-1) else level_in_shape[0]) if j==0 else cnn_depth,
                    image_shape=level_in_shape[1:],
                    cnn_depth=cnn_depth,
                    filter_size=filter_size,
                    cnn_stride=1,
                    num_layers=2,
                    padding = "same",
                    **kwargs
                    ) for j in range(layers_per_block)]
                level_net = nn.Sequential(*level_blocks)
                level_out_shape = level_blocks[-1].out_shape
            else:
                raise NotImplementedError
            level_in_shape = (level_out_shape[0], level_out_shape[1]//pooling_factor, level_out_shape[2]//pooling_factor)
            self._down_conv_blocks.append(level_net)
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(cnn_depths)-2, -1, -1):
            level_in_shape = (2 * cnn_depths[i], level_out_shape[1] * pooling_factor, level_out_shape[2] * pooling_factor)
            transpose_layer = nn.ConvTranspose2d(level_out_shape[0], cnn_depths[i], kernel_size=1, stride=pooling_factor, padding=0, output_padding=pooling_factor-1)
            if self.level_block == "conv":
                level_net = networks.ConvNet(
                    level_in_shape[0], 
                    level_in_shape[1:], 
                    [filter_size] * layers_per_block,
                    [1] * layers_per_block,
                    [cnn_depths[i]] * layers_per_block,
                    padding = "same",
                    **kwargs,
                    )
                level_out_shape = level_net.out_shape
            elif self.level_block == "res":
                level_blocks = [networks.ResBlock(
                    in_channels=level_in_shape[0]if j==0 else cnn_depths[i],
                    image_shape=level_in_shape[1:],
                    cnn_depth=cnn_depths[i],
                    filter_size=filter_size,
                    cnn_stride=1,
                    num_layers=2,
                    padding = "same",
                    **kwargs
                    ) for j in range(layers_per_block)]
                level_net = nn.Sequential(*level_blocks)
                level_out_shape = level_blocks[-1].out_shape
            else:
                raise NotImplementedError
            self._transpose_conv_layers.append(transpose_layer)
            self._up_conv_blocks.append(level_net)
        self._up_conv_blocks = nn.ModuleList(self._up_conv_blocks)
        self._transpose_conv_layers = nn.ModuleList(self._transpose_conv_layers)

        # output layer
        self._output_conv = nn.Conv2d(level_out_shape[0], out_channels, kernel_size=1, stride=1, padding="same")

    def __call__(self, x, t_embed):
        # x is (B, C, H, W)
        B, _, _, _ = x.shape
        assert t_embed.shape[0] == B and len(t_embed.shape) == 2, "t has to have shape (B, E)"
        B, E = t_embed.shape

        # downward branch
        skip_images = []
        for down_block in self._down_conv_blocks[:-1]:
            x = down_block(x)
            skip_images.append(x)
            x = F.max_pool2d(x, self.pooling_factor)
        
        # embed t TODO use a better conditioning method
        if self.encoding_dim > 0:
            _, _, latent_h, latent_w = x.shape
            t_embed = t_embed.view(B, E, 1, 1)
            t_embed = t_embed.expand(-1, -1, latent_h, latent_w)
            x = torch.cat((x, t_embed), dim = 1)

        x = self._down_conv_blocks[-1](x)

        # upward branch
        for i, (transpose_layer, up_block) in enumerate(zip(self._transpose_conv_layers, self._up_conv_blocks)):
            x = transpose_layer(x)
            x = torch.cat((x, skip_images[-(i+1)]), dim = 1)
            x = up_block(x)

        # output layer
        x = self._output_conv(x)
        return x

    def compute_pos_encodings(self, latent_shape, t):
        # t has shape (B,)
        B = t.shape[0]
        encoding_flat = self.encoding[t-1, :].view(B, self.encoding_dim, 1, 1)
        encoding = encoding_flat.expand(-1, -1, *latent_shape)
        return encoding
    
class CondUNet(nn.Module): # Conditional UNet class used for diffusion models

    def __init__(self, in_channels, out_channels, image_shape, cnn_depths, layers_per_block, filter_size, pooling_factor, cond_dim, level_block="conv", device="cpu", **kwargs):
        # length of CNN_depths determines how many levels u-net has
        super().__init__()
        self._down_conv_blocks = []
        self._up_conv_blocks = []
        self._transpose_conv_layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_shape = image_shape
        self.cnn_depths = cnn_depths
        self.filter_size = filter_size
        self.pooling_factor = pooling_factor
        self.cond_dim = cond_dim
        self.level_block = level_block

        # create downward branch
        level_in_shape = (in_channels, *image_shape)
        for i, cnn_depth in enumerate(cnn_depths):

            if self.level_block == "res":
                level_blocks = [networks.ResBlock(
                    in_channels=level_in_shape[0] if j==0 else cnn_depth,
                    image_shape=level_in_shape[1:],
                    cnn_depth=cnn_depth,
                    filter_size=filter_size,
                    cnn_stride=1,
                    num_layers=2,
                    padding = "same",
                    cond_dim = cond_dim,
                    **kwargs
                    ) for j in range(layers_per_block)]
                level_net = nn.Sequential(*level_blocks)
                level_out_shape = level_blocks[-1].out_shape
            else:
                raise NotImplementedError
            
            level_in_shape = (level_out_shape[0], level_out_shape[1]//pooling_factor, level_out_shape[2]//pooling_factor)
            self._down_conv_blocks.append(level_net)
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(cnn_depths)-2, -1, -1):
            level_in_shape = (2 * cnn_depths[i], level_out_shape[1] * pooling_factor, level_out_shape[2] * pooling_factor)
            transpose_layer = nn.ConvTranspose2d(level_out_shape[0], cnn_depths[i], kernel_size=1, stride=pooling_factor, padding=0, output_padding=pooling_factor-1)

            if self.level_block == "res":
                level_blocks = [networks.ResBlock(
                    in_channels=level_in_shape[0]if j==0 else cnn_depths[i],
                    image_shape=level_in_shape[1:],
                    cnn_depth=cnn_depths[i],
                    filter_size=filter_size,
                    cnn_stride=1,
                    num_layers=2,
                    padding = "same",
                    cond_dim = cond_dim,
                    **kwargs
                    ) for j in range(layers_per_block)]
                level_net = nn.Sequential(*level_blocks)
                level_out_shape = level_blocks[-1].out_shape
            else:
                raise NotImplementedError
            
            self._transpose_conv_layers.append(transpose_layer)
            self._up_conv_blocks.append(level_net)
        self._up_conv_blocks = nn.ModuleList(self._up_conv_blocks)
        self._transpose_conv_layers = nn.ModuleList(self._transpose_conv_layers)

        # output layer
        self._output_conv = nn.Conv2d(level_out_shape[0], out_channels, kernel_size=1, stride=1, padding="same")

    def __call__(self, x, cond_vec):
        # x is (B, C, H, W)
        B, _, _, _ = x.shape
        assert cond_vec.shape[0] == B and len(cond_vec.shape) == 2, "t has to have shape (B, E)"

        # downward branch
        skip_images = []
        for down_block in self._down_conv_blocks[:-1]:
            x,_ = down_block((x, cond_vec))
            skip_images.append(x)
            x = F.max_pool2d(x, self.pooling_factor)

        x,_ = self._down_conv_blocks[-1]((x, cond_vec))

        # upward branch
        for i, (transpose_layer, up_block) in enumerate(zip(self._transpose_conv_layers, self._up_conv_blocks)):
            x = transpose_layer(x)
            x = torch.cat((x, skip_images[-(i+1)]), dim = 1)
            x,_ = up_block((x, cond_vec))

        # output layer
        x = self._output_conv(x)
        return x