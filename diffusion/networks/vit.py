import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd
import numpy as np
import pos_encoding

class ViT(nn.Module):
    def __init__(self, num_heads, model_dim, num_layers, ff_num_layers, ff_size_factor, dropout, in_channels, out_channels, device, image_size, pos_encoding_type = "learned", patches_a_side = 16, mode = "classifier", **kwargs):
        assert (image_size // patches_a_side) * patches_a_side == image_size, "image size has to be a multiple of patches a side"
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.ff_num_layers = ff_num_layers
        self.out_channels = out_channels
        self.pos_encoding_type = pos_encoding_type
        self.device = device
        self.patch_size = image_size // patches_a_side
        self.patches_a_side = patches_a_side
        self.mode = mode
        
        self._embed = VitEmbed(patch_size=self.patch_size, patches_a_side=patches_a_side, model_dim=model_dim, in_channels=in_channels)
        self._dropout = nn.Dropout(p = dropout)
        self._attn = nn.Sequential(*[
            AttentionBlock(num_heads, model_dim, ff_num_layers, ff_size_factor, dropout) for _ in range(num_layers)
        ])
        self._layernorm = nn.LayerNorm(model_dim)
        if mode == "classifier":
            self._out_layer = VitClassifierOutput(model_dim, out_channels)
        elif mode == "linear_autoencoder":
            self._out_layer = VitLinearDecoder(model_dim, out_channels, self.patch_size, patches_a_side)
        else:
            raise NotImplementedError
        self._loss = nn.CrossEntropyLoss()
        if pos_encoding_type == "sinusoid":
            self._pos_encoding = pos_encoding.SinusoidPosEncoding(model_dim, 1)
        elif pos_encoding_type == "learned":
            self._pos_encoding = nn.Embedding(1 + patches_a_side * patches_a_side, model_dim)
        elif pos_encoding_type == "none":
            self._pos_encoding = pos_encoding.NoneEncoding()

    def __call__(self, x, *args):
        # (B, C, H, W)
        B, _, _, _ = x.shape
        embed = self._pos_encoding(torch.arange(1 + self.patches_a_side * self.patches_a_side, device=x.device)).to(x.device)
        x = torch.cat((torch.zeros((B, 1, self.model_dim), device=x.device), self._embed(x)), dim = 1) # B, T, C
        x = x + embed
        x = self._attn(x)
        x = self._layernorm(x)
        x = self._out_layer(x)
        return x
    
    def loss(self, x, y):
        logits = self(x)
        return self._loss(logits, y)

class VitClassifierOutput(nn.Module):
    def __init__(self, model_dim, num_classes):
        super().__init__()
        self.network = nn.Linear(model_dim, num_classes)
    
    def __call__(self, x):
        # x: B, T, model_dim
        return self.network(x[:, 0, :]) # use only class token at output for classification

class VitLinearDecoder(nn.Module):
    def __init__(self, model_dim, out_channels, patch_size, patches_a_side):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.patches_a_side = patches_a_side
        self.network = nn.Linear(model_dim, out_channels * patch_size * patch_size)
    
    def __call__(self, x):
        # x: B, T, model_dim
        B, T, _ = x.shape
        assert T == 1 + self.patches_a_side * self.patches_a_side, "number of tokens has to match 1+patches_per_side**2"
        x = self.network(x[:, 1:, :]) # B, T, out_channels * H * W (remove class token)
        x = x.view(B, self.patches_a_side, self.patches_a_side, self.out_channels, self.patch_size, self.patch_size) # use all tokens for generating output image
        x = torch.movedim(x, (1, 2), (-4, -2)) # B, C, H_p, H, W_p, W
        x = x.reshape(B, self.out_channels, self.patch_size * self.patches_a_side, self.patch_size * self.patches_a_side)
        return x

class VitEmbed(nn.Module):
    # patchify and embed
    def __init__(self, patch_size, patches_a_side, model_dim, in_channels):
        super().__init__()
        self.patches_a_side = patches_a_side
        self.patch_size = patch_size
        self._embed = nn.Linear(patch_size * patch_size * in_channels, model_dim)
    
    def __call__(self, x):
        B, C, _, _ = x.shape
        x = x.view(B, C, self.patches_a_side, self.patch_size, self.patches_a_side, self.patch_size)
        x = torch.movedim(x, (1, -3), (-3 ,-2)) # patchified (B, H_p, W_p, C, H, W)
        x = x.reshape(B, self.patches_a_side * self.patches_a_side, C * self.patch_size * self.patch_size)
        x = self._embed(x) # B, H_p * W_p, d
        return x

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, model_dim, ff_num_layers, ff_size_factor, dropout, att_implementation = "default"):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self._attn_dropout = nn.Dropout(p = dropout)
        self._ff_dropout = nn.Dropout(p = dropout)
        if att_implementation == "performer":
            from performer_pytorch import SelfAttention
            self._attn = SelfAttention(dim = model_dim, heads = num_heads, causal = False, dim_head=None)
        elif att_implementation == "default":
            self._attn = MultiHeadAttention(num_heads, model_dim//num_heads, model_dim//num_heads, model_dim, model_dim)
        elif att_implementation == "flash":
            self._attn = MultiHeadFlashAttention(num_heads, model_dim//num_heads, model_dim//num_heads, model_dim, model_dim)
        else:
            raise NotImplementedError
        
        ff_layers = []
        for i in range(ff_num_layers):
            in_dim = model_dim if i==0 else ff_size_factor * model_dim
            out_dim = model_dim if i==ff_num_layers-1 else ff_size_factor * model_dim
            ff_layers.append(nn.Linear(in_dim, out_dim))
            if i<(ff_num_layers-1):
                ff_layers.append(nn.ReLU())
        self._ff = nn.Sequential(*ff_layers)
        self._ln1 = nn.LayerNorm(model_dim)
        self._ln2 = nn.LayerNorm(model_dim)
    
    def __call__(self, x):
        # input: B, T, C
        # output: B, T, C
        x = x + self._attn_dropout(self._attn(self._ln1(x)))
        x = x + self._ff_dropout(self._ff(self._ln2(x)))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim, value_dim, in_dim, out_dim, mask=False):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.in_dim = in_dim
        self.mask = mask

        self._k_linear = nn.Linear(in_dim, num_heads * key_dim, bias = False)
        self._q_linear = nn.Linear(in_dim, num_heads * key_dim, bias = False)
        self._v_linear = nn.Linear(in_dim, num_heads * value_dim, bias = False)

        self._out_linear = nn.Linear(num_heads * value_dim, out_dim, bias = False)
        self._tril = None

    def __call__(self, x):
        B, T, _ = x.shape

        # preparing K, Q, V
        k = self._k_linear(x).view(B, T, self.num_heads, self.key_dim) # (B, T, n_h, d_k)
        q = self._q_linear(x).view(B, T, self.num_heads, self.key_dim) # (B, T, n_h, d_k)
        v = self._v_linear(x).view(B, T, self.num_heads, self.value_dim) # (B, T, n_h, d_v)
        
        # we want k, q reshaped so  we can matmul
        k = torch.movedim(k, 1, -1).reshape(B*self.num_heads, self.key_dim, T)
        q = torch.movedim(q, 1, -2).reshape(B*self.num_heads, T, self.key_dim)

        # computing masked scaled attention
        attn_logits = (torch.matmul(q, k)/np.sqrt(self.key_dim)).view(B, self.num_heads, T, T) # (B, n_h, T_q, T_k)
        if self.mask:
            if self._tril is None or self._tril.shape != (T,T):
                self._tril = torch.tril(torch.ones(T, T, device = attn_logits.device), diagonal=0) == 0
            attn_logits = attn_logits.masked_fill(self._tril, float('-inf')) # (B, n_h, T_q, T_k)
        attn_maps = F.softmax(attn_logits, dim = -1) # (B, n_h, T_q, T_k)
        v = torch.movedim(v, 1, 2) # (B, n_h, T, d_v)
        attn_output = torch.matmul(attn_maps, v) # (B, n_h, T_q, d_v)
        attn_output = torch.movedim(attn_output, -2, 1).reshape(B, T, self.num_heads * self.value_dim) 
        out = self._out_linear(attn_output) 
        return out # (B, T2, n_h*d_v)
    
class MultiHeadFlashAttention(nn.Module):
    def __init__(self, num_heads, key_dim, value_dim, in_dim, out_dim, mask=False):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.in_dim = in_dim
        self.mask = mask

        self._k_linear = nn.Linear(in_dim, num_heads * key_dim, bias = False)
        self._q_linear = nn.Linear(in_dim, num_heads * key_dim, bias = False)
        self._v_linear = nn.Linear(in_dim, num_heads * value_dim, bias = False)

        self._out_linear = nn.Linear(num_heads * value_dim, out_dim, bias = False)
        self._tril = None

    def __call__(self, x):
        B, T, _ = x.shape

        # preparing K, Q, V
        k = self._k_linear(x).view(B, T, self.num_heads, self.key_dim) # (B, T, n_h, d_k)
        q = self._q_linear(x).view(B, T, self.num_heads, self.key_dim) # (B, T, n_h, d_k)
        v = self._v_linear(x).view(B, T, self.num_heads, self.value_dim) # (B, T, n_h, d_v)

        k = torch.movedim(k, 1, -2) # (B, n_h, T, d_k)
        q = torch.movedim(q, 1, -2) # (B, n_h, T, d_k)
        v = torch.movedim(v, 1, -2) # (B, n_h, T, d_v)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.mask)
        
        attn_output = torch.movedim(attn_output, -2, 1)
        attn_output = attn_output.reshape(B, T, self.num_heads * self.value_dim) 
        
        out = self._out_linear(attn_output)
        return out # (B, T2, n_h*d_v)