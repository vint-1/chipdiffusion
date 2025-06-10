import torch
from torch_geometric.data import Data
import hpwl_utils

EPS = 1e-6

def hpwl_test_0():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [1, 2],
        [2, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=torch.float)
    true_hpwl = 1.6

    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    hpwl = hpwl_utils.hpwl(x, cond)
    
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"
    print("Passed hpwl 0")

def hpwl_test_1():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [1, 2],
        [2, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, -0.1, 0.05, 0.05],
    ], dtype=torch.float)
    true_hpwl = 1.8
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    hpwl = hpwl_utils.hpwl(x, cond)
    
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"
    print("Passed hpwl 1")

def hpwl_test_2():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    # (-0.65,0.05) to (0.5, -0.6)
    # (0.8, 0.8) to (0.5, -0.5), (-0.6, 0)
    true_hpwl = 1.15 + 0.65 + 1.4 + 1.3
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    hpwl = hpwl_utils.hpwl(x, cond)
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"
    print("Passed hpwl 2")

def macro_hpwl_test_0(): # same as hpwl 0, but all components are macros
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [1, 2],
        [2, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=torch.float)
    is_macros = torch.tensor([True] * x.shape[0])
    true_hpwl = 1.6

    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 0")

def macro_hpwl_test_1():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [1, 2],
        [2, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, -0.1, 0.05, 0.05],
    ], dtype=torch.float)
    is_macros = torch.tensor([True] * x.shape[0])
    true_hpwl = 1.8
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 1")

def macro_hpwl_test_2():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    is_macros = torch.tensor([True] * x.shape[0])
    # (-0.65,0.05) to (0.5, -0.6)
    # (0.8, 0.8) to (0.5, -0.5), (-0.6, 0)
    true_hpwl = 1.15 + 0.65 + 1.4 + 1.3
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 2")

def macro_hpwl_test_3():
    # test that non-macros not included
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    is_macros = torch.tensor([True, True, False])
    true_hpwl = 1.4 + 0.8
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 3")

def macro_hpwl_test_4():
    # test that net is included even if source is not macro
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    is_macros = torch.tensor([True, False, True])
    # 2nd net coords: (0.5,-0.5) and (-0.6,0)
    true_hpwl = 1.15 + 0.65 + 1.1 + 0.5
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 4")

def macro_hpwl_test_5():
    # test that net is included even if source is not macro
    # similar to test 4, but different location of non-macro
    x = torch.tensor([
        [-0.7, 0],
        [0.0, 0.0],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    is_macros = torch.tensor([True, False, True])
    # 2nd net coords: (0.5,-0.5) and (-0.6,0)
    true_hpwl = 1.15 + 0.65 + 1.1 + 0.5
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 5")

def macro_hpwl_test_6():
    # test that pin_id can deconflict pin locations
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
        [0.6, 0.6],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
        [0.1, 0.1],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [1, 3],
        [2, 0],
        [2, 1],
        [0, 1],
        [3, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, 0, 0, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=torch.float)
    edge_pin_id = torch.tensor([
        [0,1],
        [2,3],
        [2,4],
        [2,5],
        [1,0],
        [3,2],
        [4,2],
        [5,2],
    ], dtype=int)
    is_macros = torch.tensor([True, False, True, True])
    # 2nd net coords: (0.6,0.6) and (-0.6,-0.5)
    true_hpwl = 1.15 + 0.65 + 1.2 + 1.1
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros, edge_pin_id=edge_pin_id)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 6")

def macro_hpwl_test_7():
    # test that pin_id can deconflict pin locations
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
        [0.6, 0.6],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
        [0.1, 0.1],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [1, 3],
        [2, 0],
        [2, 1],
        [0, 1],
        [3, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, 0, 0, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=torch.float)
    edge_pin_id = torch.tensor([
        [0,1],
        [2,3],
        [2,4],
        [6,5],
        [1,0],
        [3,2],
        [4,2],
        [5,6],
    ], dtype=int)
    is_macros = torch.tensor([True, False, True, True])
    true_hpwl = 1.15 + 0.65 + 1.1 + 0.5
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr, is_macros=is_macros, edge_pin_id=edge_pin_id)
    macro_hpwl = hpwl_utils.macro_hpwl(x, cond)
    
    assert abs(true_hpwl - macro_hpwl) < EPS, f"True:{true_hpwl}, Given:{macro_hpwl}"
    print("Passed macro hpwl 7")

def hpwl_test_oriented_0():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    # (-0.65,0.05) to (0.5, -0.6)
    # (0.8, 0.8) to (0.5, -0.5), (-0.6, 0)
    true_hpwl = 1.15 + 0.65 + 1.4 + 1.3
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    orientation, cond_orientable = orientations.to_orientable(cond, randomize=True)
    x_orientable = torch.cat([x, orientation], dim=-1)
    hpwl = hpwl_utils.hpwl(x_orientable, cond_orientable)
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"

def hpwl_test_oriented_1():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    # (-0.65,0.05) to (0.5, -0.6)
    # (0.8, 0.8) to (0.5, -0.5), (-0.6, 0)
    true_hpwl = 1.15 + 0.65 + 1.4 + 1.3
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    orientation, cond_orientable = orientations.to_orientable(cond, randomize=False)
    x_orientable = torch.cat([x, orientation], dim=-1)
    hpwl = hpwl_utils.hpwl(x_orientable, cond_orientable)
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"

def hpwl_test_oriented_2():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    orientation = torch.tensor([
        [1, 1, -1], # 6 (FS)
        [-1, -1, 1], # 1 (E)
        [-1, 1, 1], # 3 (W)
    ], dtype=torch.float)
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    true_hpwl = 1.7 + 1.3 + 1.4
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    x_orientable = torch.cat([x, orientation], dim=-1)

    hpwl = hpwl_utils.hpwl(x_orientable, cond)
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"

def test_legality_0():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    orientation = torch.tensor([
        [1, 1, -1], # 6 (FS)
        [-1, -1, 1], # 1 (E)
        [-1, 1, 1], # 3 (W)
    ], dtype=torch.float)
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    mask = torch.tensor([
        False,
        False,
        False,
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    x_orientable = torch.cat([x, orientation], dim=-1)
    true_legality = 1

    legality = hpwl_utils.check_legality(x_orientable, x, cond, mask=mask, score=True)
    print(f"True legality:{true_legality}, Given:{legality}")

def test_legality_1():
    x = torch.tensor([
        [-0.5, 0],
        [0.0, 0],
        [0.5, 0],
    ])
    x_gt = torch.tensor([
        [-0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5],
    ])
    orientation = torch.tensor([
        [1, 1, -1], # 6 (FS)
        [-1, -1, 1], # 1 (E)
        [-1, 1, 1], # 3 (W)
    ], dtype=torch.float)
    size = torch.tensor([
        [1, 1],
        [1, 1],
        [1, 1],
    ], dtype=torch.float)
    mask = torch.tensor([
        False,
        False,
        False,
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    x_orientable = torch.cat([x, orientation], dim=-1)
    true_legality = 2/3

    legality = hpwl_utils.check_legality(x_orientable, x_gt, cond, mask=mask, score=True)
    print(f"True legality:{true_legality}, Given:{legality}")

def test_legality_2():
    x = torch.tensor([
        [-0.1, 0],
        [0.0, 0],
        [0.2, -0.3],
    ])
    x_gt = torch.tensor([
        [-0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5],
    ])
    orientation = torch.tensor([
        [1, 1, -1], # 6 (FS)
        [-1, -1, 1], # 1 (E)
        [-1, 1, 1], # 3 (W)
    ], dtype=torch.float)
    size = torch.tensor([
        [0.7, 1],
        [1, 1],
        [1, 0.2],
    ], dtype=torch.float)
    mask = torch.tensor([
        False,
        True,
        False,
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    x_orientable = torch.cat([x, orientation], dim=-1)
    true_legality = 1 - (0.7 * 0.15)/0.9

    legality = hpwl_utils.check_legality(x_orientable, x_gt, cond, mask=mask, score=True)
    print(f"True legality:{true_legality}, Given:{legality}")

def test_orientation_0():
    num_instances = 8
    x = torch.tensor([
        [-1.0 + 2*(i%(num_instances//2)+1)/(num_instances//2+1), -0.5 if i//(num_instances//2) else 0.5] for i in range(num_instances)
    ])
    size = torch.tensor([
        [0.2, 0.4] for _ in range(num_instances)
    ])
    edge_indices = torch.tensor([ 
        [i, (i+1)%num_instances] for i in range(num_instances)
    ] + [ # note that reverse edges must be included
        [(i+1)%num_instances, i] for i in range(num_instances)
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.0, 0.15, -0.08, -0.15] for _ in range(num_instances)
    ] + [ # note that reverse edges must be included
        [-0.08, -0.15, 0.0, 0.15] for _ in range(num_instances)
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    new_orientations = torch.tensor([
        [(i//4)%2, (i//2)%2, i%2] for i in range(num_instances)
    ], dtype=torch.float) * 2 - 1 
    # orientation, cond_orientable = orientations.to_orientable(cond, randomize=True)
    x_oriented = torch.cat([x, new_orientations], dim=-1)
    debug_plot(x_oriented, cond, "test_orientation_0") # This should show what each of the 8 orientations looks like

def test_orientation_1():
    num_instances = 8
    x = torch.tensor([
        [-1.0 + 2*(i%(num_instances//2)+1)/(num_instances//2+1), -0.5 if i//(num_instances//2) else 0.5] for i in range(num_instances)
    ])
    size = torch.tensor([
        [0.2, 0.4] for _ in range(num_instances)
    ])
    edge_indices = torch.tensor([ 
        [i, (i+1)%num_instances] for i in range(num_instances)
    ] + [ # note that reverse edges must be included
        [(i+1)%num_instances, i] for i in range(num_instances)
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.0, 0.15, -0.08, -0.15] for _ in range(num_instances)
    ] + [ # note that reverse edges must be included
        [-0.08, -0.15, 0.0, 0.15] for _ in range(num_instances)
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    random_orientation, cond_orientable = orientations.to_orientable(cond, randomize=True)
    x_oriented = torch.cat([x, random_orientation], dim=-1)
    debug_plot(x_oriented, cond, "test_orientation_1") # This should have randomized orientations

def test_orientation_2():
    num_instances = 8
    x = torch.tensor([
        [-1.0 + 2*(i%(num_instances//2)+1)/(num_instances//2+1), -0.5 if i//(num_instances//2) else 0.5] for i in range(num_instances)
    ])
    size = torch.tensor([
        [0.2, 0.4] for _ in range(num_instances)
    ])
    edge_indices = torch.tensor([ 
        [i, (i+1)%num_instances] for i in range(num_instances)
    ] + [ # note that reverse edges must be included
        [(i+1)%num_instances, i] for i in range(num_instances)
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.0, 0.15, -0.08, -0.15] for _ in range(num_instances)
    ] + [ # note that reverse edges must be included
        [-0.08, -0.15, 0.0, 0.15] for _ in range(num_instances)
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    random_orientation, cond_orientable = orientations.to_orientable(cond, randomize=True)
    x_oriented = torch.cat([x, random_orientation], dim=-1)
    debug_plot(x_oriented, cond_orientable, "test_orientation_2") # This should have reconstructed orientations

def test_orientation_3():
    num_instances = 8
    x = torch.tensor([
        [-1.0 + 2*(i%(num_instances//2)+1)/(num_instances//2+1), -0.5 if i//(num_instances//2) else 0.5] for i in range(num_instances)
    ])
    size = torch.tensor([
        [0.2, 0.4] for _ in range(num_instances)
    ])
    edge_indices = torch.tensor([ 
        [i, (i+1)%num_instances] for i in range(num_instances)
    ] + [ # note that reverse edges must be included
        [(i+1)%num_instances, i] for i in range(num_instances)
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.0, 0.15, -0.08, -0.15] for _ in range(num_instances)
    ] + [ # note that reverse edges must be included
        [-0.08, -0.15, 0.0, 0.15] for _ in range(num_instances)
    ], dtype=torch.float)
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    new_orientations = torch.tensor([
        [(i//4)%2, (i//2)%2, i%2] for i in range(num_instances)
    ], dtype=torch.float) * 2 - 1 
    
    fixed_cond = orientations.to_fixed(new_orientations, cond)
    debug_plot(x, fixed_cond, "test_orientation_3") 

def test_hpwl_guidance_0():
    x = torch.tensor([
        [-0.7, 0],
        [0.8, 0.8],
        [0.5, -0.5],
    ])
    size = torch.tensor([
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.2],
    ])
    edge_indices = torch.tensor([ # note that reverse edges must be included
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 0],
        [2, 1],
        [0, 1],
    ], dtype=torch.long).T
    edge_attr = torch.tensor([
        [0.05, 0.05, 0, -0.1],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0],
        [0, -0.1, 0.05, 0.05],
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
    ], dtype=torch.float)
    # (-0.65,0.05) to (0.5, -0.6)
    # (0.8, 0.8) to (0.5, -0.5), (-0.6, 0)
    true_hpwl = 1.15 + 0.65 + 1.4 + 1.3
    
    cond = Data(size, edge_index=edge_indices, edge_attr=edge_attr)
    hpwl = guidance.hpwl_guidance_potential(x, cond)
    assert abs(true_hpwl - hpwl) < EPS, f"True:{true_hpwl}, Given:{hpwl}"
    print("Passed hpwl guidance 0")

def test_clustering():
    # DEBUGGING TODO REMOVE
    device = "cuda"
    x = 1
    cond = 2
    model = 3
    # image = utils.visualize_placement(x, cond, plot_edges=True, img_size=(1024, 1024))
    # utils.debug_plot_img(image, "ibm0_unclustered")
    
    cluster_cond, cluster_x = hpwl_utils.cluster(cond, 512, placements=x)
    image = hpwl_utils.visualize_placement(cluster_x, cluster_cond, plot_edges=True, img_size=(1024, 1024))
    hpwl_utils.debug_plot_img(image, "ibm0_clustered")
    
    with torch.no_grad():
        cluster_x = cluster_x.unsqueeze(dim=0).to(device=device)
        cluster_cond = cluster_cond.to(device=device)
        placement_x, intermediates = model.reverse_samples(1, cluster_x, cluster_cond, intermediate_every = 0)
    image = hpwl_utils.visualize_placement(placement_x[0], cluster_cond, plot_edges=True, img_size=(1024, 1024))
    hpwl_utils.debug_plot_img(image, "ibm0_placed512")
    
    unclustered_placement = hpwl_utils.uncluster(cluster_cond, placement_x)
    cond = cond.to(device=device)
    image = hpwl_utils.visualize_placement(unclustered_placement[0], cond, plot_edges=True, img_size=(1024, 1024))
    hpwl_utils.debug_plot_img(image, "ibm0_placed512_unclustered")
    import ipdb; ipdb.set_trace()

def debug_plot(x, cond, name="test_img"):
    img = hpwl_utils.visualize_placement(x, cond, plot_edges = True)
    hpwl_utils.debug_plot_img(img, name)

if __name__=="__main__":
    hpwl_test_0()
    hpwl_test_1()
    hpwl_test_2()
    macro_hpwl_test_0()
    macro_hpwl_test_1()
    macro_hpwl_test_2()
    macro_hpwl_test_3()
    macro_hpwl_test_4()
    macro_hpwl_test_5()
    macro_hpwl_test_6()
    macro_hpwl_test_7()
    # hpwl_test_oriented_0()
    # hpwl_test_oriented_1()
    # hpwl_test_oriented_2()
    # test_legality_0()
    # test_legality_1()
    # test_legality_2()
    # test_hpwl_guidance_0()
    # test_orientation_0()
    # test_orientation_1()
    # test_orientation_2()
    # test_orientation_3()