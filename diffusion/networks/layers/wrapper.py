import torch
import torch.nn as nn

class BatchWrapper(nn.Module):
    
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, edge_index, edge_attr=None, **kwargs):
        # process x, edge_index, edge_attr
        # x: (B, V, F)
        # edge_index: (2, E)
        # edge attributes: (E, F)
        B, V, F = x.shape
        _, E = edge_index.shape
        
        x_unbatched = x.reshape(B * V, F)

        if edge_attr is not None:
            edge_attr_unbatched = edge_attr.view(1, *edge_attr.shape).expand(B, -1, -1)
            edge_attr_unbatched = edge_attr_unbatched.reshape(B * E, -1)
        else:
            edge_attr_unbatched = None
        
        edge_index_unbatched = edge_index.movedim(-1, 0).unsqueeze(dim=0).expand(B, E, 2)
        edge_index_offset = torch.arange(0, V*B, V, device=edge_index.device, dtype=edge_index.dtype).view(B, 1, 1)
        edge_index_unbatched = edge_index_unbatched + edge_index_offset
        edge_index_unbatched = edge_index_unbatched.reshape(B * E, 2).movedim(0, -1)
        
        output_unbatched = self.net(x_unbatched, edge_index_unbatched, edge_attr=edge_attr_unbatched, **kwargs) # (B*V, F)

        # reshape output
        output = output_unbatched.view(B, V, -1)
        return output