import torch
import torch.nn.functional as F
import torch_geometric.nn as tgn

BLOCK_SIZE = 16384 # 8192 # 16384

def legality_guidance_potential(x_hat, cond, softmax_factor=10.0, mask=None):
    """
    Differentiable function for computing legality guidance potential
    Inputs:
    - x_hat (B, V, 2)
    - cond is pytorch data object
    """
    B, V, D = x_hat.shape
    sizes = cond.x.expand(B, *cond.x.shape) # (B, V, D)

    # compute energy h. we use convention that higher h = less favorable ie. force = -grad(h)
    x_1 = x_hat.view(B, V, 1, D)
    x_2 = x_hat.view(B, 1, V, D).detach()
    size_1 = sizes.view(B, V, 1, D)
    size_2 = sizes.view(B, 1, V, D)
    delta = torch.abs(x_1 - x_2) - ((size_1 + size_2)/2) # (B, V1, V2, D)
    # softmax and max both work
    l = torch.sum(F.softmax(delta * softmax_factor, dim=-1) * delta, dim=-1, keepdim=True)
    h = (F.relu(-l)**2) / 4

    # calculate boundary term
    h_bound = (F.relu(torch.abs(x_hat) + sizes/2 - 1) ** 2)/2 # (B, V, D)

    # mask out objects where mask=True and self-collisions
    mask_square = (1-torch.eye(V, dtype=h.dtype, device=h.device)).view(1, V, V, 1) # ignore self-collision
    if mask is not None:
        inv_mask = ~mask
        mask_square = mask_square * inv_mask.view(1, 1, V, 1) * inv_mask.view(1, V, 1, 1)
        h_bound = inv_mask * h_bound
    
    # weight forces by size of instances
    mass_1 = torch.exp(torch.mean(torch.log(sizes), dim=-1, keepdim=True)).unsqueeze(dim=-1) # (B, V1, 1, 1)
    mass_2 = mass_1.view(B, 1, V, 1) # (B, 1, V2, 1)
    h = h * mask_square * ((mass_2)/(mass_1 + mass_2)) # (B, V1, V2, D)
    h_bound = h_bound

    # compute forces
    h_dims = list(range(1, len(h.shape)))
    h_bound_dims = list(range(1, len(h_bound.shape)))
    h_total = h.sum(dim = h_dims) + h_bound.sum(dim = h_bound_dims)
    return h_total

def legality_guidance_potential_tiled(x_hat, cond, softmax_factor=10.0, mask=None, block_size=BLOCK_SIZE):
    """
    Differentiable function for computing legality guidance potential
    This is a tiled, memory-constrained version of the above
    Inputs:
    - x_hat (B, V, 2)
    - cond is pytorch data object

    NOTE this function performs backward passes wrt x_hat to save memory
    Output:
    - Detached legality potential
    """
    B, V, D = x_hat.shape

    h_bound = legality_potential_boundary(x_hat, cond, mask=mask)
    h = 0
    for start_i in range(0, V, block_size):
        end_i = min(start_i + block_size, V)
        for start_j in range(0, V, block_size):
            end_j = min(start_j + block_size, V)
            h_current = legality_potential_tile(
                x_hat[:, start_i:end_i, :], 
                x_hat[:, start_j:end_j, :], 
                cond.x[start_i:end_i, :],
                cond.x[start_j:end_j, :],
                start_i == start_j,
                mask_1 = None if mask is None else mask[:, start_i:end_i, :],
                mask_2 = None if mask is None else mask[:, start_j:end_j, :],
                softmax_factor = softmax_factor,
                )
            # backward pass to save memory
            h_current.backward()
            h = h + h_current.detach()
    h_bound.backward()
    h_total = h + h_bound.detach()
    return h_total

def legality_potential_tile(x_hat_1, x_hat_2, size_1, size_2, is_diagonal, mask_1 = None, mask_2 = None, softmax_factor=10.0):
    """
    Differentiable function for computing legality guidance potential
    Inputs:
    - x_hat (B, V, 2)
    - size (V, 2)
    - masks are (1, V, 2) or (B, V, 2)
    - is_diagonal specifies if self-collisions should be masked out
    TODO make it so that we don't have to duplicate computations for the lower triangle
    NOTE this is possible by multiplying m1*m2/(m1+m2) and scaling gradients w.r.t m1 before optimizer.step
    NOTE if this is to be done, we also have to be careful of the diagonal and to not detach x_2
    """
    B, V_1, D = x_hat_1.shape
    B_2, V_2, D_2 = x_hat_2.shape
    assert (B == B_2) and (D == D_2), "input x must have same batch and feature dimensions"

    # compute energy h. we use convention that higher h = less favorable ie. force = -grad(h)
    x_1 = x_hat_1.view(B, V_1, 1, D)
    x_2 = x_hat_2.view(B, 1, V_2, D).detach()
    size_1 = size_1.expand(B, *size_1.shape).view(B, V_1, 1, D)
    size_2 = size_2.expand(B, *size_2.shape).view(B, 1, V_2, D)
    delta = torch.abs(x_1 - x_2) - ((size_1 + size_2)/2) # (B, V1, V2, D)
    # softmax and max both work
    l = torch.sum(F.softmax(delta * softmax_factor, dim=-1) * delta, dim=-1, keepdim=True)
    h = (F.relu(-l)**2) / 4

    # mask out objects where mask=True and self-collisions
    if is_diagonal:
        mask_square = (1-torch.eye(n=V_1, m=V_2, dtype=h.dtype, device=h.device)).view(1, V_1, V_2, 1) # ignore self-collision
    else:
        mask_square = 1
    if (mask_1 is not None) and (mask_2 is not None):
        inv_mask_1 = ~mask_1
        inv_mask_2 = ~mask_2
        mask_square = mask_square * inv_mask_1.view(1, V_1, 1, 1) * inv_mask_2.view(1, 1, V_2, 1)
    
    # weight forces by size of instances
    mass_1 = torch.exp(torch.mean(torch.log(size_1), dim=-1, keepdim=True)) # (B, V1, 1, 1)
    mass_2 = torch.exp(torch.mean(torch.log(size_2), dim=-1, keepdim=True)) # (B, 1, V2, 1)
    h = h * mask_square * ((mass_2)/(mass_1 + mass_2)) # (B, V1, V2, D)

    # compute forces
    h_dims = list(range(1, len(h.shape)))
    h_tile = h.sum(dim = h_dims)
    return h_tile

def legality_potential_boundary(x_hat, cond, mask=None):
    """
    Differentiable function for computing boundary-enforcement term of legality guidance potential
    Inputs:
    - x_hat (B, V, 2)
    - cond is pytorch data object
    """
    B, V, D = x_hat.shape
    sizes = cond.x.view(1, V, D)

    # calculate boundary term
    h_bound = (F.relu(torch.abs(x_hat) + sizes/2 - 1) ** 2)/2 # (B, V, D)

    # mask out objects where mask=True and self-collisions
    if mask is not None:
        inv_mask = ~mask
        h_bound = inv_mask * h_bound

    # compute forces
    h_bound_dims = list(range(1, len(h_bound.shape)))
    return h_bound.sum(dim = h_bound_dims)

def hpwl_guidance_potential(x, cond, pin_map=None, pin_offsets=None, pin_edge_index=None, hpwl_net=None):
    """
    Differentiable function for computing hpwl
    Inputs:
    - x (B, V, 2)
    - cond is pytorch data object with edge_index (2, E) and edge_attr (E, 4)
    - pin map, offsets, edge index, and hpwl_net are optional variables that should be cached per-netlist
    """
    # compute netlist-level info if cached version not provided
    if pin_map is None or pin_offsets is None or pin_edge_index is None:
        pin_map, pin_offsets, pin_edge_index = compute_pin_map(cond)
    if hpwl_net is None:
        hpwl_net = HPWL()
    
    # compute and return hpwl
    hpwl = hpwl_net(x, pin_map, pin_offsets, pin_edge_index)
    return hpwl

def hpwl_square_guidance_potential(x, cond, pin_map=None, pin_offsets=None, pin_edge_index=None, hpwl_net=None):
    """
    Differentiable function for computing hpwl-based potential, using square of distance
    Inputs:
    - x (B, V, 2)
    - cond is pytorch data object with edge_index (2, E) and edge_attr (E, 4)
    - pin map, offsets, edge index, and hpwl_net are optional variables that should be cached per-netlist
    """
    # compute netlist-level info if cached version not provided
    if pin_map is None or pin_offsets is None or pin_edge_index is None:
        pin_map, pin_offsets, pin_edge_index = compute_pin_map(cond)
    if hpwl_net is None:
        hpwl_net = HPWL()
    
    # compute and return hpwl
    hpwl = hpwl_net(x, pin_map, pin_offsets, pin_edge_index, net_aggr = "none") # (B, P)
    hpwl_potential = ((hpwl ** 2)/2).sum(dim=-1)
    return hpwl_potential

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
    