import torch
import torch.nn as nn
import torch.functional as F
from .mlp import FiLM
from .vit import AttentionBlock
import pos_encoding

class GraphTransformer(nn.Module):
    # Graph Transformer using positional encodings to embed graph structure
    def __init__(
            self, 
            in_node_features,
            out_node_features, 
            edge_features,
            num_heads,
            model_dim,
            num_layers,
            encoding_dim, # this is for encoding t
            pos_encoding_dim,
            pos_encoding_type = "laplacian",
            ff_num_layers = 2,
            ff_size_factor = 4,
            extra_node_features = 2,
            dropout=0.0, 
            device="cpu",
            residual = False,
            **kwargs,
            ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.edge_features = edge_features
        self.device = device
        self.model_dim = model_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.extra_node_features = extra_node_features
        self.residual = residual
        if self.residual:
            assert self.in_node_features == self.out_node_features, "in and out features must be equal for residual network"

        # network layers
        if pos_encoding_type == "laplacian":
            self._pos_encoding = pos_encoding.LaplacianEncoding(self.pos_encoding_dim) # TODO
        elif pos_encoding_type == "none":
            self._pos_encoding = pos_encoding.NoneEncoding((1, 1, self.pos_encoding_dim))
        else:
            raise NotImplementedError
        # 2 extra dimensions from graph node attr
        self._in_layer = nn.Linear(self.in_node_features + self.pos_encoding_dim + self.extra_node_features, model_dim)
        self._t_in_layer = nn.Linear(encoding_dim, self.model_dim)
        self._dropout = nn.Dropout(p = dropout)
        self._attn = nn.Sequential(*[
            AttentionBlock(
                num_heads, 
                model_dim, 
                ff_num_layers, 
                ff_size_factor, 
                dropout
            ) for _ in range(num_layers)
        ])
        self._layernorm = nn.LayerNorm(model_dim)
        self._out_layer = nn.Linear(model_dim, out_node_features)

    def __call__(self, x, cond, t_embed):
        # x: (B, V, C)
        # t_embed: (B, d_model)
        B, V, C = x.shape
        x0 = x
        t_embed = self._t_in_layer(t_embed).unsqueeze(dim=1)
        pos_embed = self._pos_encoding(cond).to(x.device).unsqueeze(dim=0)
        pos_embed = torch.broadcast_to(pos_embed, (B, V, self.pos_encoding_dim)) # (1, V, d_enc)
        extra_node_attr = cond.x.unsqueeze(dim=0)
        extra_node_attr = torch.broadcast_to(extra_node_attr, (B, V, self.extra_node_features))
        x = torch.cat((x, extra_node_attr, pos_embed), dim=-1) # (B, V, d)
        x = self._in_layer(x) # (B, V, d_model)
        x = torch.cat((t_embed, x), dim=1) # (B, V+1, d_model)
        x = self._attn(x)[:, 1:, :]
        x = self._layernorm(x)
        x = self._out_layer(x)
        if self.residual:
            x = x + x0
        return x