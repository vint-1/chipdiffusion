import torch
import torch.nn.functional as F
import torch_geometric.nn as tgn
from torch_geometric.data import Data

class HPWL(tgn.MessagePassing):
    def __init__(self):
        super().__init__(aggr="max", flow="target_to_source") 
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, pin_map, pin_offsets, pin_edge_index, net_aggr = "sum", raw_output = False):
        # x is (B, V, 2) matrix defining the placement
        # pin_map is (P) tensor such that pin_map[pin_idx] = object_idx
        # pin_offsets is (P, 2) tensor
        # pin_edge_index has shape (2, E) note that this has to preprocessed
        global_pin_position = pin_offsets + x[..., pin_map, :] # (B, P, 2)
        # Start propagating messages.
        net_maxmin = F.relu(self.propagate(pin_edge_index, x=global_pin_position)) # (B, P, 4)
        if raw_output:
            return net_maxmin
        net_hpwl = torch.sum(net_maxmin, dim=-1)
        if net_aggr == "sum":
            hpwl = torch.sum(net_hpwl, dim=-1)
        elif net_aggr == "mean":
            hpwl = torch.mean(net_hpwl, dim=-1)
        elif net_aggr == "none":
            hpwl = net_hpwl
        else:
            raise NotImplementedError
        return hpwl

    def message(self, x_i, x_j):
        # x_i and x_j has shape (B, E, 2)
        delta = x_j - x_i
        delta_combined = torch.cat((delta, -delta), dim=-1) # (B, E, 4)
        return delta_combined
    
class MacroHPWL(tgn.MessagePassing):
    def __init__(self):
        super().__init__(aggr="max", flow="target_to_source") 
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, pin_map, pin_offsets, pin_edge_index, is_macro, net_aggr = "sum", raw_output = False):
        # x is (B, V, 2) matrix defining the placement
        # pin_map is (P) tensor such that pin_map[pin_idx] = object_idx
        # pin_offsets is (P, 2) tensor
        # pin_edge_index has shape (2, E) note that this has to preprocessed
        global_pin_position = pin_offsets + x[..., pin_map, :] # (B, P, 2)
        pin_is_macro = is_macro[..., pin_map, None]
        # Start propagating messages.
        net_maxmin = self.propagate(pin_edge_index, x=global_pin_position, is_macro=pin_is_macro) # (B, P, 4)
        net_maxmin[net_maxmin==float('-inf')] = 0
        if raw_output:
            return net_maxmin
        net_hpwl = torch.sum(net_maxmin, dim=-1)
        if net_aggr == "sum":
            hpwl = torch.sum(net_hpwl, dim=-1)
        elif net_aggr == "mean":
            hpwl = torch.mean(net_hpwl, dim=-1)
        elif net_aggr == "none":
            hpwl = net_hpwl
        else:
            raise NotImplementedError
        return hpwl

    def message(self, x_i, x_j, is_macro_i, is_macro_j):
        # x_i and x_j has shape (B, E, 2)
        # is_macro_i and is_macro_j have shape (B, E, 1)
        data_i = torch.cat((x_i, -x_i), dim=-1) # (B, E, 4)
        data_j = torch.cat((x_j, -x_j), dim=-1) # (B, E, 4)
        
        # ignore non-macros
        data_i_masked = torch.where(is_macro_i, data_i, float('-inf'))
        data_j_masked = torch.where(is_macro_j, data_j, float('-inf'))
        data_combined_masked = torch.maximum(data_i_masked, data_j_masked) # (B, E, 4)
        return data_combined_masked

def compute_pin_map(cond):
    """
    Computes tensors needed for computing hpwl efficiently \\
    Returns:
    - Pin map: (P) tensor such that pin_map[pin_idx] = object_idx
    - Pin offsets: (P, 2) tensor with offsets for each pin
    - Pin_edge_index: edge_index, except using pin indices instead of object indices 
        (so pin_map[pin_edge_index] == edge_index_unique)
    """
    _, E = cond.edge_index.shape
    assert E % 2 == 0, "cond edge index assumed to contain forward and reverse edges"
    edge_index_unique = cond.edge_index[:, :E//2].T # (E, 2)
    edge_attr_unique = cond.edge_attr[:E//2, :]
    if "edge_pin_id" in cond:
        edge_pin_id_unique = cond.edge_pin_id[:E//2, :]

    # note: we convert to double to avoid float roundoff error for >17M edges
    if "edge_pin_id" in cond:
        sources = torch.cat((
            edge_index_unique[:,0:1].double(), 
            edge_attr_unique[:,0:2].double(), 
            edge_pin_id_unique[:,0:1].double(),
            ), dim=1)
        dests = torch.cat((
            edge_index_unique[:,1:2].double(), 
            edge_attr_unique[:,2:4].double(),
            edge_pin_id_unique[:,1:2].double(),
            ), dim=1)
    else:
        sources = torch.cat((
            edge_index_unique[:,0:1].double(), 
            edge_attr_unique[:,0:2].double()
            ), dim=1)
        dests = torch.cat((
            edge_index_unique[:,1:2].double(), 
            edge_attr_unique[:,2:4].double()
            ), dim=1)
    edge_endpoints = torch.cat((sources, dests), dim=0) # (2E, 3/4)
    
    # get unique pins
    pin_info, pin_inverse_index = torch.unique(edge_endpoints, return_inverse=True, dim=0) # (E_u, 3), (2E)
    pin_edge_index = pin_inverse_index.view(2, E//2)
    pin_map = pin_info[:, 0].type(cond.edge_index.dtype)
    pin_offsets = pin_info[:, 1:3].type(cond.edge_attr.dtype)
    return pin_map, pin_offsets, pin_edge_index

def hpwl(samples, cond_val, normalized_hpwl = True):
    """ 
    Computes HPWL
    samples is (V, 2) tensor with 2D coordinates OR (V, 2+3) with orientation in float bits describing placement of center of instances
    cond is pytorch geometric Data object with the following:
    - edge_index (2, E)
    - edge_attr (E, 4) tensor describing pin locations, measured relative to center of instance
    """
    # net format
    # [inst_id, driver_pin_x, driver_pin_y]: list of absolute sink pin locations

    assert len(samples.shape) == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    assert samples.shape[1] == 2, "orientations not supported"
    samples = samples[:,:2].cpu()

    nets = {}

    unique_edges = len(cond_val.edge_attr)//2
    for ids, pins in zip(cond_val.edge_index.T[:unique_edges], cond_val.edge_attr[:unique_edges]):
        u_id, v_id = ids
        ux, uy, vx, vy = pins

        # key is the component id and pin position
        key = str([u_id, ux, uy])

        u_loc = (samples[u_id][0].item() + ux.item(), samples[u_id][1].item() + uy.item())
        v_loc = (samples[v_id][0].item() + vx.item(), samples[v_id][1].item() + vy.item())
        nets[key] = nets.get(key, u_loc) + v_loc
    
    # half perimeter = (max x - min x) + (max y - min y)
    norm_hpwl = sum([(max(n[::2]) - min(n[::2])) + (max(n[1::2]) - min(n[1::2])) for n in nets.values()])
    if normalized_hpwl:
        return norm_hpwl
    else:
        x_scale = (cond_val.chip_size[2] - cond_val.chip_size[0])/2 # because chip is from [-1, 1] when normed
        y_scale = (cond_val.chip_size[3] - cond_val.chip_size[1])/2
        rescaled_hpwl = sum([x_scale * (max(n[::2]) - min(n[::2])) + y_scale * (max(n[1::2]) - min(n[1::2])) for n in nets.values()])
        return norm_hpwl, rescaled_hpwl

def hpwl_fast(x, cond, normalized_hpwl = True):
    """
    Returns hpwl computed using custon GNN trick
    If not normalized_hpwl, will return both normalized HPWL, as well as rescaled HPWL (using original units)
    """
    hpwl_net = HPWL()
    pin_map, pin_offsets, pin_edge_index = compute_pin_map(cond)
    hpwl_net = hpwl_net(x, pin_map, pin_offsets, pin_edge_index, net_aggr="sum", raw_output = (not normalized_hpwl))
    if normalized_hpwl:
        return hpwl_net.item() # output is hpwl, no additional processing needed
    else:
        x_scale = (cond.chip_size[2] - cond.chip_size[0])/2 # because chip is from [-1, 1] when normed
        y_scale = (cond.chip_size[3] - cond.chip_size[1])/2
        scale_factor = torch.tensor([[x_scale, y_scale, x_scale, y_scale]]).to(device = hpwl_net.device)
        rescaled_hpwl = (scale_factor * hpwl_net).sum(dim=-1).sum(dim=-1) 
        norm_hpwl = hpwl_net.sum(dim=-1).sum(dim=-1)
        return norm_hpwl.item(), rescaled_hpwl.item()
    
def macro_hpwl(x, cond, normalized_hpwl = True):
    """ 
    Computes macro HPWL
    samples is (V, 2) tensor with 2D coordinates OR (V, 2+3) with orientation in float bits describing placement of center of instances
    cond is pytorch geometric Data object with the following:
    - edge_index (2, E)
    - edge_attr (E, 4) tensor describing pin locations, measured relative to center of instance
    - is_macros must be in cond
    """
    hpwl_net = MacroHPWL()
    pin_map, pin_offsets, pin_edge_index = compute_pin_map(cond)
    hpwl_net = hpwl_net(x, pin_map, pin_offsets, pin_edge_index, cond.is_macros, net_aggr="sum", raw_output = (not normalized_hpwl))
    if normalized_hpwl:
        return hpwl_net.item() # output is hpwl, no additional processing needed
    else:
        x_scale = (cond.chip_size[2] - cond.chip_size[0])/2 # because chip is from [-1, 1] when normed
        y_scale = (cond.chip_size[3] - cond.chip_size[1])/2
        scale_factor = torch.tensor([[x_scale, y_scale, x_scale, y_scale]]).to(device = hpwl_net.device)
        rescaled_hpwl = (scale_factor * hpwl_net).sum(dim=-1).sum(dim=-1) 
        norm_hpwl = hpwl_net.sum(dim=-1).sum(dim=-1)
        return norm_hpwl.item(), rescaled_hpwl.item()