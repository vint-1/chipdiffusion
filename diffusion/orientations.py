# Helper functions for working with orientable (rotate + flip) macros
# TODO For guidance to work, we need to_fixed to be differentiable (only this fn is needed for guidance)
# NOTE Because we only require to_fixed to be defined on discrete bits, there are lots of ways to interpolate in a differentiable way...
# NOTE So we probably have to come up with something clever

import torch
from torch_geometric.data import Data

ORIENTATION_BITS = torch.tensor([
    [-1,-1,-1],
    [-1,-1, 1],
    [-1, 1,-1],
    [-1, 1, 1],
    [ 1,-1,-1],
    [ 1,-1, 1],
    [ 1, 1,-1],
    [ 1, 1, 1],
],
    dtype = torch.float
)

TRANSFORM = torch.tensor([
    [[1,0],[0,1]],
    [[0,1],[-1,0]],
    [[-1,0],[0,-1]],
    [[0,-1],[1,0]],
    [[-1,0],[0,1]],
    [[0,-1],[-1,0]],
    [[1,0],[0,-1]],
    [[0,1],[1,0]],
],
    dtype = torch.float
)
TRANSFORM_INV = torch.inverse(TRANSFORM)

def to_orientable(cond, randomize=True):
    """
    Inputs:
    
    cond - Data object with:
        x(V,2) as instance size (as-placed);
        edge_index(2,E);
        edge_attr(E,4+) with 2D coordinates of pins relative to instance center (as-placed);
        is_ports(V) of booleans;

    randomize - if set to True, will pick random orientations for each instance, otherwise will use N (for debugging)
    
    For each instance, randomly picks one of 8 possible orientations as 0-orientation,
    then converts x and cond to an orientable placement, relative to the picked 0-orientations.
    Instances masked by is_ports will still be orientable.
    Note that this is the inverse of to_fixed.
    
    Returns:
    
    orientation - tensor(V, 3) 3 bits {-1, 1} describing orientation relative to 0-orientation

    cond - Data object with:
        x(V,2) as instance size, measured in 0-orientation;
        edge_index(2,E);
        edge_attr(E,4+) with 2D coordinates of pins relative to instance center, measured in 0-orientation;
        is_ports(V) of booleans;
    """
    V, _ = cond.x.shape
    device = cond.x.device
    transform_inv = TRANSFORM_INV.to(device=device)

    orientation_int = torch.randint(8, size = (V,), device=device) if randomize else torch.zeros((V,), device=device, dtype=torch.int)
    orientation = ORIENTATION_BITS.to(device=device)[orientation_int] # (V, 3)
    
    size_transforms = torch.abs(transform_inv[orientation_int]) # (V, 2, 2)
    output_instance_sizes = torch.einsum("vij,vj->vi", size_transforms, cond.x)

    output_cond = cond.clone("edge_attr")

    u_transforms = transform_inv[orientation_int[cond.edge_index[0,:]]] # (E, 2, 2)
    v_transforms = transform_inv[orientation_int[cond.edge_index[1,:]]] # (E, 2, 2)
    output_cond.edge_attr[:,:2] = torch.einsum("eij,ej->ei", u_transforms, output_cond.edge_attr[:,:2])
    output_cond.edge_attr[:,2:4] = torch.einsum("eij,ej->ei", v_transforms, output_cond.edge_attr[:,2:4])

    output_cond.x = output_instance_sizes
    return orientation, output_cond

def to_fixed(orientation, cond): # TODO figure out how to do batching
    """
    Inputs:

    orientation - tensor(V, 3) 3 bits {-1, 1} describing orientations relative to 0-orientation
    
    cond - Data object with:
        x(V,2) as instance size, measured in 0-orientation;
        edge_index(2,E);
        edge_attr(E,4+) with 2D coordinates of pins relative to instance center, measured in 0-orientation;
        is_ports(V) of booleans;
    
    Uses orientations specified in placement to generate instance sizes and pin positions.
    Performs discretization of orientation bits if necessary
    Note that this is the inverse of to_orientable.
    
    Returns:
    
    cond - Data object with:
        x(V,2) as instance size (as-placed);
        edge_index(2,E);
        edge_attr(E,4+) with 2D coordinates of pins relative to instance center (as-placed);
        is_ports(V) of booleans;
    """
    V, _ = cond.x.shape
    device = cond.x.device
    transform = TRANSFORM.to(device=device)

    orientation_int = discretize_orientation(orientation)

    size_transforms = torch.abs(transform[orientation_int]) # (V, 2, 2)
    output_instance_sizes = torch.einsum("vij,vj->vi", size_transforms, cond.x)

    output_cond = cond.clone("edge_attr")

    u_transforms = transform[orientation_int[cond.edge_index[0,:]]] # (E, 2, 2)
    v_transforms = transform[orientation_int[cond.edge_index[1,:]]] # (E, 2, 2)
    output_cond.edge_attr[:,:2] = torch.einsum("eij,ej->ei", u_transforms, output_cond.edge_attr[:,:2])
    output_cond.edge_attr[:,2:4] = torch.einsum("eij,ej->ei", v_transforms, output_cond.edge_attr[:,2:4])

    output_cond.x = output_instance_sizes
    return output_cond

def relative_orientation(base_orientation, new_orientation):
    """
    Finds orientation s.t. to_fixed(orientation, cond) == to_fixed(new_orientation, to_orientable(cond)[1])
    Where base_orientation is to_orientable(cond)[0]
    Inputs: (B, V, 3) in continuous space
    Outputs: (B, V, 3) bits in float dtype describing orientation
    """
    B, V, _ = base_orientation.shape
    device = base_orientation.device
    base_int = discretize_orientation(base_orientation) # A_1
    new_int = discretize_orientation(new_orientation) # A_2
    
    transform_inv = TRANSFORM_INV.to(device=device)
    transform = TRANSFORM.to(device=device)

    a_1_inv = transform_inv[base_int] # (B, V, 2, 2)
    a_2 = transform[new_int] # (B, V, 2, 2)

    relative_transform = torch.einsum("bvij,bvjk->bvik", a_2, a_1_inv).view(1, B, V, transform.shape[1]*transform.shape[2])
    transform_offset = torch.abs(relative_transform - transform.view(transform.shape[0], 1, 1, transform.shape[1]*transform.shape[2])) # (8, B, V, 4)
    transform_distances = transform_offset.sum(dim=-1) # (8, B, V)

    output_orientation_int = torch.argmin(transform_distances, dim=0) # (B, V)
    output_orientation = ORIENTATION_BITS.to(device=device)[output_orientation_int] # (V, 3)
    return output_orientation


def discretize_orientation(orientation): 
    """
    Takes orientation(B, V, 3) or (V, 3) in continuous space and outputs integer describing orientation
    Returns (B, V) or (V) tensor, depending on input
    """
    V, bits = orientation.shape[-2:]
    assert bits == 3, "orientation must have 3 bits"
    device = orientation.device

    orientation = orientation.contiguous() #  TODO fix warnings about non-contiguous data?
    orientation_discrete = torch.bucketize(orientation, torch.tensor([0], device=device)) # (B, V, 3)
    orientation_int = orientation_discrete[..., 0]*4 + orientation_discrete[..., 1]*2 + orientation_discrete[..., 2]
    return orientation_int

