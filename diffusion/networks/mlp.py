import pos_encoding
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_layers, model_width, in_size, out_size, skip = False, layernorm = False, **kwargs):
        super().__init__()
        assert (not skip) or (in_size == out_size), "input and output dimensions must be equal for skip connection"
        self.num_layers = num_layers
        self.model_width = model_width
        self.in_size = in_size
        self.out_size = out_size
        layers = []
        for i in range(num_layers):
            inputs = in_size if i == 0 else model_width
            outputs = out_size if i == num_layers-1 else model_width
            layers.append(nn.Linear(inputs, outputs))
            layers.append(nn.ReLU())
        layers = layers[:-1] # remove final activation layer
        self.skip = skip
        self.layernorm = layernorm
        self._nn = nn.Sequential(*layers)
        self._ln = nn.LayerNorm(in_size)
    
    def __call__(self, x):
        # x is (..., D)
        in_shape = x.shape
        output = self._nn(self._ln(x.view(-1, in_shape[-1]).view(*in_shape))) if self.layernorm else self._nn(x)
        return x + output if self.skip else output

class ConditionalMLP(nn.Module):
    encodings = {"sinusoid": pos_encoding.get_positional_encodings}

    def __init__(self, num_layers, model_width, in_size, out_size, encoding_type, encoding_dim, max_diffusion_steps, device = "cpu", **kwargs):
        super().__init__()
        self.encoding_type = encoding_type
        self.encoding = ConditionalMLP.encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.mlp = MLP(num_layers, model_width, in_size + encoding_dim, out_size, **kwargs)

    def __call__(self, x, t):
        # t is a vector with shape (B)
        B = x.shape[0]
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        t_encoded = self.encoding[t-1]
        x_original = x.view(B, -1)
        x = torch.cat((x_original, t_encoded), dim = -1)
        return x_original + self.mlp(x)

class ResidualMLP(nn.Module):
    encodings = {"sinusoid": pos_encoding.get_positional_encodings}

    def __init__(self, num_blocks, layers_per_block, model_width, in_size, out_size, encoding_type, encoding_dim, max_diffusion_steps, device = "cpu", **kwargs):
        # several residual blocks, each an MLP
        super().__init__()
        self.encoding_type = encoding_type
        self.encoding = ResidualMLP.encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.num_blocks = num_blocks
        blocks = []
        self._encoder = nn.Linear(in_size + encoding_dim, model_width)
        self._decoder = nn.Linear(model_width, out_size)
        for i in range(num_blocks):
            new_block = MLP(layers_per_block, model_width, model_width, model_width, skip = True, layernorm = True, **kwargs)
            blocks.append(new_block)
        self._model = nn.Sequential(*blocks)

    def __call__(self, x, t):
        # t is a vector with shape (B)
        B = x.shape[0]
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        t_encoded = self.encoding[t-1]
        x = x.view(B, -1)
        x = torch.cat((x, t_encoded), dim = -1)
        x_original = self._encoder(x)
        x = self._model(x_original)
        x = self._decoder(x_original + x)
        return x
    
class FiLM(nn.Module):
    def __init__(self, cond_dim, input_dim, channel_axis=0):
        # NOTE channel_axis excludes batch dimension
        # so for images with (C, H, W), channel_axis should be 0
        super().__init__()
        self._mult_proj = nn.Linear(cond_dim, input_dim) 
        self._add_proj = nn.Linear(cond_dim, input_dim)
        self.channel_axis = channel_axis
    
    def __call__(self, x, cond):
        # x has (B, input_dim, ...) or (B, ..., input_dim)
        # cond has (B, cond_dim)
        mult = self._mult_proj(cond)
        add = self._add_proj(cond)
        # sort out the shapes
        if len(x.shape) > 2:
            B, cond_dim = mult.shape
            expand_dims = [1]*(len(x.shape)-1)
            expand_dims[self.channel_axis] = cond_dim
            mult = mult.view(B, *expand_dims)
            add = add.view(B, *expand_dims)
        return mult * x + add