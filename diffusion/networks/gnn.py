import torch
import torch_geometric as tg
import torch_geometric.nn as tgn
import torch.nn as nn
import torch.functional as F
import numpy as np
from .mlp import FiLM, MLP
from .vit import AttentionBlock
import networks.layers as layers

def get_conv_layer(layer_type, in_channels, out_channels, edge_features, **layer_kwargs):
    layer_fns = {
        "gcn": tgn.GCNConv,
        "sage": tgn.SAGEConv,
        "gin": tgn.GINConv,
        "transformer": tgn.TransformerConv, # TODO figure out why this doesn't work
        "custom_transformer": layers.CustomTransformerConv,
        "gated": layers.GatedGraphConv,
        "gat": tgn.GATv2Conv,
    }
    if layer_type == "gin":
        layer_params = {
            "nn": MLP(
                num_layers = layer_kwargs.get("nn_num_layer", 2), 
                model_width = layer_kwargs.get("nn_hidden_width", out_channels),
                in_size = in_channels, 
                out_size = out_channels,
            ),
            **layer_kwargs
        }
    elif layer_type == "gated":
        assert in_channels <= out_channels
        layer_params = {
            "out_channels": out_channels,
            **layer_kwargs
        }
    elif layer_type == "gat" or layer_type == "transformer":
        if layer_kwargs.get("concat", True):
            assert (out_channels % layer_kwargs.get("heads", 1) == 0), "out channels must be divisible by number of heads in GAT"
            channel_divisor = layer_kwargs.get("heads", 1)
        else:
            channel_divisor = 1
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels // channel_divisor,
            "edge_dim": edge_features,
            **layer_kwargs
        }
    elif layer_type == "custom_transformer":
        if layer_kwargs.get("concat", True):
            assert (out_channels % layer_kwargs.get("heads", 1) == 0), "out channels must be divisible by number of heads in GAT"
            channel_divisor = layer_kwargs.get("heads", 1)
        else:
            channel_divisor = 1
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels // channel_divisor,
            "edge_dim": edge_features,
            **layer_kwargs
        }
    else:
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            **layer_kwargs
        }
    layer = layer_fns[layer_type](**layer_params)
    if layer_type in ["gat", "transformer", "custom_transformer"]:
        # use batch wrapper since tgn does not support batched dim
        layer = layers.BatchWrapper(layer)
    return layer

def accepts_edge_attr(layer):
    # checks if layer takes edge attribute as input
    return isinstance(layer, layers.BatchWrapper)

class GConvLayer(nn.Module):
    def __init__(self, in_node_features, out_node_features):
        super().__init__()
        self._layer = tgn.GCNConv(in_node_features, out_node_features)
    
    def forward(self, x_in):
        x, data, _ = x_in
        edge_index = data.edge_index
        return self._layer(x, edge_index)

class LinearEncoderLayer(nn.Module):
    def __init__(self, in_node_features, out_node_features, input_encoding_dim=0, mask_key=None, device="cpu"):
        MAX_FREQ = 100
        super().__init__()
        assert input_encoding_dim % 2 == 0, "input encoding dimension must be even"
        mask_features = 1 if mask_key is not None else 0
        self._layer = nn.Linear(in_node_features + mask_features, out_node_features)
        self._encoding_layer = nn.Linear(in_node_features * input_encoding_dim, out_node_features) if input_encoding_dim>0 else None
        self.input_encoding_dim = input_encoding_dim
        self.input_encoding_freqs = torch.exp(
            np.log(MAX_FREQ) * torch.arange(0, self.input_encoding_dim // 2, dtype=torch.float32, device=device) / (self.input_encoding_dim // 2)
        ).view(1, 1, 1 , self.input_encoding_dim // 2)
        self.mask_key = mask_key
    
    def forward(self, x, cond_data, t_embed):
        node_data = cond_data.x
        node_data = node_data.view(1, *node_data.shape).expand(x.shape[0], -1, -1)
        spatial_input = torch.concatenate((x, node_data), dim=-1)
        if self.mask_key is not None:
            node_mask = cond_data[self.mask_key]
            node_mask = node_mask.float().view(1, *node_mask.shape, 1).expand(x.shape[0], -1, 1)
            proj_input = torch.concatenate((spatial_input, node_mask), dim=-1)
        else:
            proj_input = spatial_input
        output = self._layer(proj_input)
        if self._encoding_layer is not None:
            input_encodings = self.get_input_encoding(spatial_input)
            input_encodings_proj = self._encoding_layer(input_encodings)
            output = output + input_encodings_proj
        return output

    def get_input_encoding(self, spatial_input):
        # spatial_input: (B, V, D)
        B, V, D = spatial_input.shape
        
        theta = spatial_input.unsqueeze(dim=-1) * self.input_encoding_freqs
        embedding = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) # (B, V, D, E)
        embedding = embedding.view(B, V, D * self.input_encoding_dim)

        return embedding


class LinearDecoderLayer(nn.Module):
    def __init__(self, in_node_features, out_node_features):
        super().__init__()
        self._layer = nn.Linear(in_node_features, out_node_features)
    
    def forward(self, x, cond_data, t_embed):
        return self._layer(x)

class ResGNNBlock(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, 
            cond_node_features, 
            edge_features, 
            num_layers, 
            encoding_dim,
            residual=True, 
            norm=True,
            dropout=0.0, 
            conv_params={"layer_type": "gcn"}, 
            device="cpu", 
            **kwargs
        ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.residual = residual
        if residual:
            assert in_node_features == out_node_features, "input and output features must be equal to perform residual connection"
        self._gconv_layers = nn.ModuleList()
        self._lnorm_layers = nn.ModuleList()
        self._linear_layers = nn.ModuleList()

        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim>0 else None
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features, 
                out_channels=hidden_node_features,
                edge_features=edge_features,
                **conv_params
                ))
            self._lnorm_layers.append(nn.LayerNorm(hidden_node_features))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
            
        self.use_edge_attr = accepts_edge_attr(self._gconv_layers[0])
        # self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    def forward(self, x, data, t): # data is conditioning info
        B, V, F = x.shape
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        x_skip = x
        x = torch.cat((x, cond_x), dim=-1)
        for i, (lnorm, linear, conv) in enumerate(zip(self._lnorm_layers[:-1], self._linear_layers[:-1], self._gconv_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            x = conv(x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else conv(x, edge_index)
            x = self._nonlinear(x)
            x = lnorm(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self._gconv_layers[-1](x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else self._gconv_layers[-1](x, edge_index)
        if (not self._cond_layer is None):
            x = self._cond_layer(x, t)
        x = self._nonlinear(x)
        x = self._lnorm_layers[-1](x)
        x = self._linear_layers[-1](x)
        if self.residual:
            x = x + x_skip 
        return x 

class AttGNNBlock(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, 
            cond_node_features,
            attention_extra_features, 
            edge_features, 
            num_layers, 
            encoding_dim, 
            residual=True, 
            norm=True, 
            dropout=0.0, 
            conv_params={"layer_type": "gcn"}, 
            device="cpu", 
            **kwargs
        ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_extra_features = attention_extra_features
        self.edge_features = edge_features
        self.residual = residual
        if residual:
            assert in_node_features == out_node_features, "input and output features must be equal to perform residual connection"
        self._gconv_layers = []
        self._att_extra_input_embed_layers = []
        self._attention_layers = []
        self._lnorm_layers = []
        self._linear_layers = []

        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim>0 else None
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features, 
                out_channels=hidden_node_features,
                edge_features=edge_features,
                **conv_params
                ))
            self._att_extra_input_embed_layers.append(nn.Linear(cond_node_features + attention_extra_features, hidden_node_features))
            self._attention_layers.append(AttentionBlock(
                kwargs["num_heads"], 
                hidden_node_features, 
                kwargs["ff_num_layers"], 
                kwargs["ff_size_factor"], 
                dropout, 
                att_implementation = kwargs["att_implementation"]
            ))
            self._lnorm_layers.append(nn.LayerNorm(hidden_node_features))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
            # self._linear_layers.append(MLP(mlp_num_layers, att_model_size * mlp_size_factor, att_model_size, out_features, skip = att_model_size==out_features, layernorm = True))
        
        self.use_edge_attr = accepts_edge_attr(self._gconv_layers[0])
        self._gconv_layers = nn.ModuleList(self._gconv_layers)
        self._att_extra_input_embed_layers = nn.ModuleList(self._att_extra_input_embed_layers)
        self._attention_layers = nn.ModuleList(self._attention_layers)
        self._lnorm_layers = nn.ModuleList(self._lnorm_layers)
        self._linear_layers = nn.ModuleList(self._linear_layers)
        # self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    def forward(self, x, data, t, att_extra_input = None): # data is conditioning info
        B, V, F = x.shape
        assert att_extra_input is None or att_extra_input.shape[-1] == self.attention_extra_features, "extra attention features must have right shape"
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        x_skip = x
        x = torch.cat((x, cond_x), dim=-1)
        x_att_features = torch.cat((cond_x, att_extra_input), dim=-1) if att_extra_input is not None else cond_x
        for i, (lnorm, linear, conv, attention, att_input_embed_layer) in enumerate(zip(self._lnorm_layers[:-1], self._linear_layers[:-1], self._gconv_layers[:-1], self._attention_layers[:-1], self._att_extra_input_embed_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            x = conv(x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else conv(x, edge_index)
            x = self._nonlinear(x)
            att_extra_embedded = att_input_embed_layer(x_att_features)
            x = x + att_extra_embedded
            # x = torch.cat((x, x_att_features), dim=-1)
            x = attention(x)
            x = lnorm(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self._gconv_layers[-1](x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else self._gconv_layers[-1](x, edge_index)
        if (not self._cond_layer is None):
            x = self._cond_layer(x, t)
        x = self._nonlinear(x)
        att_extra_embedded = self._att_extra_input_embed_layers[-1](x_att_features)
        x = x + att_extra_embedded
        # x = torch.cat((x, x_att_features), dim=-1)
        x = self._attention_layers[-1](x)
        x = self._lnorm_layers[-1](x)
        x = self._linear_layers[-1](x)
        if self.residual:
            x = x + x_skip 
        return x

class ResGNN(nn.Module):
    blocks = {"res": ResGNNBlock, "att": AttGNNBlock}
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_size, 
            hidden_node_features, 
            cond_node_features, 
            edge_features, 
            layers_per_block, 
            encoding_dim, 
            dropout=0.0, 
            device="cpu", 
            block_type="res", 
            **kwargs
            ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features

        self._gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        if self.use_enc:
            self._gnn_blocks.append(GConvLayer(in_node_features, hidden_size))
        for i, hidden_node_size in enumerate(hidden_node_features):
            self._gnn_blocks.append(ResGNN.blocks[block_type](
                in_node_features=hidden_size,
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=layers_per_block,
                encoding_dim=encoding_dim,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
                **kwargs,
            ))
        if self.use_enc:
            self._gnn_blocks.append(GConvLayer(hidden_size, out_node_features))
        self._network = nn.Sequential(*self._gnn_blocks)
        print("ENCODER USED IN RESGNN", self.use_enc)

    def forward(self, x, cond, t_embed):
        x_skip = x
        x,_,_ = self._network((x, cond, t_embed))
        return (x + x_skip if self.use_enc else x)

class AttGNN(nn.Module):
    # Same as ResGNN, but with attention layer in between resBlocks
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_size, 
            hidden_node_features,
            attention_node_features, 
            cond_node_features,
            edge_features, 
            layers_per_block, 
            t_encoding_dim,
            conv_params,
            mlp_num_layers,
            mlp_size_factor,
            input_encoding_dim=0,
            dir_att_input=False,
            mask_key=None,
            dropout=0.0,
            device="cpu",
            **kwargs, # should contain attention parameters
            ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_node_features = attention_node_features
        self.attention_extra_features = in_node_features # TODO fix case for dir_att_input=False
        self.edge_features = edge_features
        self.dir_att_input = dir_att_input
        self.mask_key = mask_key
        self.device = device

        gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        if self.use_enc:
            gnn_blocks.append(LinearEncoderLayer(
                in_node_features + cond_node_features, 
                hidden_size, 
                mask_key=mask_key, 
                input_encoding_dim=input_encoding_dim,
                device=device,
            ))
        for i, (hidden_node_size, attention_node_size) in enumerate(zip(hidden_node_features, attention_node_features)):
            gnn_blocks.append(ResGNNBlock(
                in_node_features=hidden_size,
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=layers_per_block,
                encoding_dim=t_encoding_dim,
                conv_params=conv_params,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
            ))
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                gnn_blocks.append(MLP(
                    mlp_num_layers, 
                    mlp_size_factor * hidden_size, 
                    hidden_size, 
                    hidden_size, 
                    skip = True, 
                    layernorm = True,
                ))
            if attention_node_size > 0:
                gnn_blocks.append(AttGNNBlock(
                    in_node_features=hidden_size,
                    out_node_features=hidden_size,
                    hidden_node_features=attention_node_size,
                    cond_node_features=cond_node_features,
                    attention_extra_features=self.attention_extra_features,
                    edge_features=edge_features,
                    num_layers=1,
                    encoding_dim=t_encoding_dim,
                    conv_params=conv_params,
                    residual=True,
                    norm=True,
                    dropout=dropout,
                    device=device,
                    **kwargs,
                ))
            else:
                gnn_blocks.append(ResGNNBlock(
                    in_node_features=hidden_size,
                    out_node_features=hidden_size,
                    hidden_node_features=hidden_node_size,
                    cond_node_features=cond_node_features,
                    edge_features=edge_features,
                    num_layers=1,
                    encoding_dim=t_encoding_dim,
                    conv_params=conv_params,
                    residual=True,
                    norm=True,
                    dropout=dropout,
                    device=device,
                ))
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                gnn_blocks.append(MLP(
                    mlp_num_layers, 
                    mlp_size_factor * hidden_size, 
                    hidden_size, 
                    hidden_size, 
                    skip = True, 
                    layernorm = True,
                ))
        if self.use_enc:
            gnn_blocks.append(LinearDecoderLayer(hidden_size, out_node_features))
            if self.in_node_features != self.out_node_features:
                self._skip_linear = nn.Linear(in_node_features, self.out_node_features)
        self._gnn_blocks = nn.ModuleList(gnn_blocks)
        print("ENCODER USED IN ATTGNN", self.use_enc)

    def forward(self, x, cond, t_embed):
        with torch.autocast(device_type=self.device):
            x_skip = x
            for block in self._gnn_blocks:
                if isinstance(block, AttGNNBlock): # include attention conditioning
                    att_input = x_skip if self.dir_att_input else x
                    x = block(x, cond, t_embed, att_extra_input=att_input)
                elif isinstance(block, MLP):
                    x = block(x)
                else:
                    x = block(x, cond, t_embed)
            if self.use_enc:
                if self.in_node_features != self.out_node_features:
                    x_skip = self._skip_linear(x_skip)
                x = x + x_skip
        return x

class GraphUNet(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, # list
            cond_node_features, 
            edge_features, 
            blocks_per_level, # list
            layers_per_block,
            level_block="res", 
            device="cpu", 
            **kwargs
        ):
        # length of CNN_depths determines how many levels u-net has
        super().__init__()
        self._down_conv_blocks = []
        self._up_conv_blocks = []
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.cond_node_features = cond_node_features
        self.blocks_per_level = blocks_per_level
        self.layers_per_block = layers_per_block
        self.level_block = level_block
        self.device=device

        # create downward branch
        for i, (hidden_size, num_blocks) in enumerate(zip(hidden_node_features, blocks_per_level)):
            level_in_size = in_node_features if i==0 else hidden_node_features[i-1]
            if self.level_block == "res":
                level_in_layer = GConvLayer(level_in_size, hidden_size)
                level_blocks = [ResGNNBlock(
                    in_node_features = hidden_size, 
                    out_node_features = hidden_size, 
                    hidden_node_features = hidden_size, 
                    cond_node_features = cond_node_features, 
                    edge_features = edge_features, 
                    num_layers = layers_per_block,
                    device = device,
                    **kwargs
                    ) for _ in range(num_blocks)]
                if i == len(hidden_node_features)-1:
                    level_blocks.append(GConvLayer(hidden_size, level_in_size))
                level_net = nn.Sequential(level_in_layer, *level_blocks)
            else:
                raise NotImplementedError
            
            self._down_conv_blocks.append(level_net)
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(hidden_node_features)-2, -1, -1):
            level_in_size = 2 * hidden_node_features[i]
            level_out_size = hidden_node_features[i-1] if i>0 else out_node_features
            hidden_size = hidden_node_features[i]
            num_blocks = blocks_per_level[i]
            if self.level_block == "res":
                level_in_layer = GConvLayer(level_in_size, hidden_size)
                level_blocks = [ResGNNBlock(
                    in_node_features = hidden_size, 
                    out_node_features = hidden_size, 
                    hidden_node_features = hidden_size, 
                    cond_node_features = cond_node_features, 
                    edge_features = edge_features, 
                    num_layers = layers_per_block,
                    device = device,
                    **kwargs
                    ) for _ in range(layers_per_block)]
                level_out_layer = GConvLayer(hidden_size, level_out_size)
                level_net = nn.Sequential(level_in_layer, *level_blocks, level_out_layer)
            else:
                raise NotImplementedError
            self._up_conv_blocks.append(level_net)
        self._up_conv_blocks = nn.ModuleList(self._up_conv_blocks)

    def __call__(self, x, data, t_enc):
        # x is (B, V, F)
        B, _, _ = x.shape
        assert t_enc.shape[0] == B and len(t_enc.shape) == 2, "t has to have shape (B, E)"
        x_skip = x

        # downward branch
        skip_images = []
        for down_block in self._down_conv_blocks[:-1]:
            x, _, _ = down_block((x, data, t_enc))
            skip_images.append(x)

        x, _, _ = self._down_conv_blocks[-1]((x, data, t_enc))

        # upward branch
        for i, up_block in enumerate(self._up_conv_blocks):
            x = torch.cat((x, skip_images[-(i+1)]), dim = -1)
            x, _, _ = up_block((x, data, t_enc))
        
        return x + x_skip