# TODO things like plotting
import os
import torch
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import sklearn.cluster
from PIL import Image, ImageDraw
import pickle
import csv

def get_hyperedges(cond_val):
    # TODO actually, we should use pin IDs
    hyp_e = {}
    insts = cond_val.edge_index.T[:len(cond_val.edge_attr)//2]
    inst_pins = cond_val.edge_attr[:len(cond_val.edge_attr)//2]

    # lib_components = {k:v for v, k in enumerate(torch.unique(torch.flatten(cond_val.edge_index)))}
    # get_key = lambda name, x, y: str([name, x, y])

    # We add 1 to all ids, such that ids are 1 -> num(nodes) inclusive
    for i, p in zip(insts, inst_pins):
        u, v = i.tolist()
        ux, uy, vx, vy = p.tolist()
        if (u, ux, uy) in hyp_e:
            hyp_e[(u, ux, uy)] += [v + 1]
        else:
            hyp_e[(u, ux, uy)] = [u + 1, v + 1]
    return hyp_e

def get_hyperedges_2(cond_val):
    # TODO test for edge cases like hyperedges with >2 vertices
    # exclude macros from hyperedges
    hyp_e = {}
    _, E = cond_val.edge_index.shape
    insts = cond_val.edge_index.T[:E//2, :]
    inst_pins = cond_val.edge_pin_id[:E//2, :] if "edge_pin_id" in cond_val else cond_val.edge_attr[:E//2, :] 

    # We add 1 to all ids, such that ids are 1 -> num(nodes) inclusive
    # exclude macros and ports from hyperedges
    for i, p in zip(insts, inst_pins):
        u, v = i.tolist()
        u_id = tuple(p[:len(p)//2].tolist())
        if (u, u_id) not in hyp_e:
            hyp_e[(u, u_id)] = []
        if not (cond_val.is_macros[u] or cond_val.is_ports[u]):
            hyp_e[(u, u_id)] += [u+1]
        if not (cond_val.is_macros[v] or cond_val.is_ports[v]):
            hyp_e[(u, u_id)] += [v+1]
    
    # remove hyperedges with <= 1 nodes
    output_hyperedges = {}
    for k, v in hyp_e.items():
        if len(v) > 1:
            output_hyperedges[k] = v
    return output_hyperedges

def cluster(input_cond, num_clusters, ubfactor=5, temp_dir = "logs/temp", verbose = False, placements = None, algorithm = None):
    """
    placements should be (B, V, 2) tensor
    Note: outputs will be on same device as inputs
    """

    # move to cpu for clustering
    placements_device = placements.device
    cond_device = input_cond.x.device
    placements.to(device = "cpu")
    input_cond.to(device = "cpu")
    _, E = input_cond.edge_index.shape
    assert E % 2 == 0, "cond edge index assumed to contain forward and reverse edges"

    hyp_e = get_hyperedges(input_cond)
    try:
        os.makedirs(temp_dir)
    except FileExistsError:
        pass
    if placements is None:
        placements = torch.zeros_like(input_cond.x).unsqueeze(dim=0) # (B, V, 2)
    filename = os.path.join(temp_dir, 'edges.txt')
    # input file
    # first line is number of hyperedges and number of vertices
    # i th line (excluding comment lines) contains the vertices that are included in the (i−1)th hyperedge
    with open(filename, 'w') as fp:
        fp.write(f'{len(hyp_e)} {max([n for k in hyp_e for n in hyp_e[k]])}\n')
        for k in hyp_e:
            fp.write(' '.join(map(str, hyp_e[k])) + '\n')

    # call hmetis and parse output
    # shmetis HGraphFile Nparts UBfactor
    subprocess.run(['./shmetis', filename, str(num_clusters), str(int(ubfactor))], capture_output = not verbose)

    with open(f'{filename}.part.{num_clusters}', 'r') as fp:
        assigned_parts = list(map(int, fp.readlines()))
    assert len(assigned_parts) == input_cond.x.shape[0], f"error parsing shmetis output. expected lines {input_cond.x.shape[0]} but got {len(assigned_parts)}"

    # create new cond_val
    B, _, _ = placements.shape
    components = list(zip(input_cond.x, assigned_parts, input_cond.is_ports, input_cond.is_macros, range(len(input_cond.is_ports))))
    data_x = []
    data_ports = [torch.tensor(False) for _ in range(num_clusters)]
    data_macros = [torch.tensor(False) for _ in range(num_clusters)]
    cluster_area = [torch.tensor(0, dtype = input_cond.x.dtype) for _ in range(num_clusters)]
    cluster_centroid = torch.zeros((B, num_clusters, 2), dtype=placements.dtype, device=placements.device)
    non_clustered_ids = [] # (V_unclustered) list containing IDs in output (clustered) cond
    # so non_clustered_id[instance_idx] = cluster_idx
    for i, (size, part, port, macro, _) in enumerate(components):
        if port or macro:
            non_clustered_ids.append(len(data_macros))
            data_x.append(size)
            data_ports.append(port)
            data_macros.append(macro)
        else:
            non_clustered_ids.append(part)
            obj_area = size[0]*size[1]
            cluster_area[part] += obj_area
            cluster_centroid[:, part, :] += obj_area * placements[:, i, :]
    macro_port_placements = placements[:, torch.logical_or(input_cond.is_macros, input_cond.is_ports), :]
    cluster_placements = cluster_centroid / torch.stack(cluster_area, dim=0).view((1, num_clusters, 1)).to(device=cluster_centroid.device)
    output_placements = torch.cat((cluster_placements, macro_port_placements), dim = 1)

    # ensuring a 1:1 aspect ratio
    data_x_macro = [torch.tensor([math.sqrt(cluster_area[part]), math.sqrt(cluster_area[part])], dtype=input_cond.x.dtype) for part in range(num_clusters)]
    data_x = data_x_macro + data_x

    # new edges should specify the index of the cluster
    new_id_raw = lambda u : components[u][1] if not(components[u][2] or components[u][3]) else non_clustered_ids[u]
    new_id = lambda x : new_id_raw(x.item())

    # generate new pin ids
    if "edge_pin_id" in input_cond:
        edge_index_unique = input_cond.edge_index[:, :E//2].T # (E, 2)
        edge_pin_id_unique = input_cond.edge_pin_id[:E//2, :] # (E, 2)
        sources = torch.cat((
            edge_index_unique[:,0:1].double(), 
            edge_pin_id_unique[:,0:1].double(),
            ), dim=1)
        dests = torch.cat((
            edge_index_unique[:,1:2].double(),
            edge_pin_id_unique[:,1:2].double(),
            ), dim=1)
        edge_endpoints = torch.cat((sources, dests), dim=0) # (2E, 2)
        _, global_pin_ids = torch.unique(edge_endpoints, return_inverse=True, dim=0) # (E_u, 3), (2E)
        global_pin_ids = global_pin_ids.view(2, E//2)
        reverse_pin_ids = torch.cat((global_pin_ids[1:2, :], global_pin_ids[0:1, :]), dim=0)
        global_pin_ids = torch.cat((global_pin_ids, reverse_pin_ids), dim=1) # (2, 2E)
        output_edge_pin_id = torch.stack([
            torch.tensor([u_id, v_id], dtype=input_cond.edge_pin_id.dtype)
            for (u, v), (u_id, v_id) in zip(input_cond.edge_index.T, global_pin_ids.T)
            if new_id(u) != new_id(v)])

    # ensuring all ports are at the center of the macro
    # port_macro = [data_x_macro[part][0]/2 for part in range(num_clusters)]
    new_port = lambda u, e: 0 if u < num_clusters else e
    try: # DEBUGGING TODO
        output_edge_index = torch.stack([
            torch.tensor([new_id(u), new_id(v)], dtype=input_cond.edge_index.dtype) 
            for (u, v), e in zip(input_cond.edge_index.T, input_cond.edge_attr)
            if new_id(u) != new_id(v)]).movedim(-1, 0)
        output_edge_attr = torch.stack([
            torch.tensor([new_port(new_id(u), e[0]), new_port(new_id(u), e[1]), new_port(new_id(v), e[2]), new_port(new_id(v), e[3])], dtype=input_cond.edge_attr.dtype)
            for (u, v), e in zip(input_cond.edge_index.T, input_cond.edge_attr)
            if new_id(u) != new_id(v)])
        output_cluster_map = torch.tensor(non_clustered_ids, dtype = torch.int)
    except:
        import ipdb; ipdb.set_trace()

    if (output_placements.isinf().any().item() or output_placements.isnan().any().item()):
        empty_clusters = set(assigned_parts) - set(non_clustered_ids)
        zero_area_clusters = set([i for i, area in enumerate(cluster_area) if area.item() <= 1e-12])
        print("WARNING: Empty cluster(s) detected: ", input_cond, empty_clusters, zero_area_clusters)
        # deal with clusters with no standard cells
        for empty_cluster_idx in sorted(empty_clusters, reverse=True):
            # remove entry for empty cluster from masks and placement
            data_x.pop(empty_cluster_idx)
            data_ports.pop(empty_cluster_idx)
            data_macros.pop(empty_cluster_idx)
            output_placements = torch.cat((output_placements[:, :empty_cluster_idx, :], output_placements[:, (empty_cluster_idx+1):, :]), dim=1)
            # remap edge_index and cluster_map
            output_edge_index = output_edge_index - (output_edge_index > empty_cluster_idx).int()
            output_cluster_map = output_cluster_map - (output_cluster_map > empty_cluster_idx).int()
        assert not (output_placements.isinf().any().item() or output_placements.isnan().any().item()), "nans still present in data"
        assert output_edge_index.max() < len(data_x), "edge indices mapped incorrectly while removing empty clusters"
        assert output_cluster_map.max() < len(data_x), "output cluster maps generated incorrectly while removing empty clusters"

    output_x = torch.stack(data_x, dim=0)
    output_ports = torch.stack(data_ports, dim=0)
    output_macros = torch.stack(data_macros, dim=0)

    output_cond = Data(
        x = output_x,
        is_ports = output_ports,
        is_macros = output_macros,
        edge_index = output_edge_index,
        edge_attr = output_edge_attr,
        cluster_map = output_cluster_map,
    )
    if "edge_pin_id" in input_cond:
        output_cond.edge_pin_id = output_edge_pin_id
    # (shallow) copy over other attributes in cond
    for k in input_cond.keys():
        if not k in output_cond:
            output_cond[k] = input_cond[k]

    # return to original device
    placements.to(device = placements_device)
    input_cond.to(device = cond_device)
    
    return output_cond.to(device = cond_device), output_placements.to(device = placements_device)

def run_hmetis(input_cond, num_clusters, algorithm = "hmetis", ubfactor=5, temp_dir = "logs/temp", verbose = False):
    """
    algorithm is one of "hmetis", "shmetis", "khmetis"
    """
    hyp_e = get_hyperedges_2(input_cond)
    try:
        os.makedirs(temp_dir)
    except FileExistsError:
        pass
    
    salt = np.random.randint(1e12) # so we can run operations in parallel
    hmetis_input_file = os.path.join(temp_dir, f'edges-{salt}.txt')
    # input file
    # first line is number of hyperedges and number of vertices
    # i th line (excluding comment lines) contains the vertices that are included in the (i−1)th hyperedge
    with open(hmetis_input_file, 'w') as fp:
        fp.write(f'{len(hyp_e)} {input_cond.x.shape[0]}\n')
        for k in hyp_e:
            fp.write(' '.join(map(str, hyp_e[k])) + '\n')

    # call hmetis and parse output
    # shmetis HGraphFile Nparts UBfactor
    if algorithm == "shmetis":
        subprocess.run([
            './shmetis', 
            hmetis_input_file, 
            str(num_clusters), 
            str(int(ubfactor)),
            ], capture_output = not verbose)
    elif algorithm == "hmetis":
        # hmetis HGraphFile Nparts UBfactor Nruns CType Rtype Vcycle Reconst debuglevel
        nruns = 25 # 10 is shmetis default
        ctype = 1 # shmetis default
        rtype = 1 # shemtis default
        vcycle = 3 # 1 is shmetis default; consider using 3 for slower but better
        reconst = 1 # 0 is shmetis default; 1 is reconstruct partial hyperedges
        dbglvl = 0 # 0 is no debugging; 31 is for everything
        subprocess.run([
            './hmetis', 
            hmetis_input_file, 
            str(num_clusters), 
            str(int(ubfactor)), 
            str(nruns), 
            str(ctype), 
            str(rtype),
            str(vcycle),
            str(reconst),
            str(dbglvl),
            ], capture_output = not verbose)
    elif algorithm == "khmetis":
        # khmetis HGraphFile Nparts UBfactor Nruns CType Otype Vcycle debuglevel
        nruns = 10 # 10
        ctype = 1 # shmetis default
        otype = 2 # 2 is minimize sum of ext. degrees
        vcycle = 3 # 1 is shmetis default; consider using 3 for slower but better
        dbglvl = 0 # 0 is no debugging; 31 is for everything
        subprocess.run([
            './khmetis', 
            hmetis_input_file, 
            str(num_clusters), 
            str(int(ubfactor)), 
            str(nruns), 
            str(ctype), 
            str(otype),
            str(vcycle),
            str(dbglvl),
            ], capture_output = not verbose)
    else:
        raise NotImplementedError

    hmetis_output_file = f'{hmetis_input_file}.part.{num_clusters}'
    with open(hmetis_output_file, 'r') as fp:
        assigned_parts = list(map(int, fp.readlines()))
    assert len(assigned_parts) == input_cond.x.shape[0], f"error parsing shmetis output. expected lines {input_cond.x.shape[0]} but got {len(assigned_parts)}"
    
    # clean up hmetis i/o files
    os.remove(hmetis_input_file)
    os.remove(hmetis_output_file)
    
    return assigned_parts

def oracle_cluster(cond, num_clusters, placements):
    """
    placements should be (1, V, 2) tensor
    Note: outputs will be on same device as inputs
    """
    B, V, _ = placements.shape
    assert B == 1, "oracle clustering cannot handle batch size > 1"
    placements = placements[0, :, :]
    weights = (~(cond.is_macros | cond.is_ports)).float()
    kmeans = sklearn.cluster.KMeans(num_clusters, init="k-means++", n_init="auto").fit(placements, sample_weight=weights)
    cluster_labels = kmeans.labels_
    return cluster_labels

def cluster_2(input_cond, num_clusters, ubfactor=5, temp_dir = "logs/temp", verbose = False, placements = None, algorithm = "hmetis"):
    """
    placements should be (B, V, 2) tensor
    Note: outputs will be on same device as inputs
    """

    # move to cpu for clustering
    if placements is None:
        placements = torch.zeros_like(input_cond.x).unsqueeze(dim=0) # (B, V, 2)
    placements_device = placements.device
    cond_device = input_cond.x.device
    placements.to(device = "cpu")
    input_cond.to(device = "cpu")
    _, E = input_cond.edge_index.shape
    assert E % 2 == 0, "cond edge index assumed to contain forward and reverse edges"

    # perform clustering
    if algorithm == "oracle":
        assigned_parts = oracle_cluster(input_cond, num_clusters, placements)
    elif algorithm in ("shmetis", "hmetis", "khmetis"):
        assigned_parts = run_hmetis(input_cond, num_clusters, algorithm = algorithm, ubfactor=ubfactor, temp_dir = temp_dir, verbose = verbose)
    else:
        raise NotImplementedError

    # create new cond_val
    B, _, _ = placements.shape
    components = list(zip(input_cond.x, assigned_parts, input_cond.is_ports, input_cond.is_macros, range(len(input_cond.is_ports))))
    data_x = []
    data_ports = [torch.tensor(False) for _ in range(num_clusters)]
    data_macros = [torch.tensor(False) for _ in range(num_clusters)]
    cluster_area = [torch.tensor(0, dtype = input_cond.x.dtype) for _ in range(num_clusters)]
    cluster_centroid = torch.zeros((B, num_clusters, 2), dtype=placements.dtype, device=placements.device)
    non_clustered_ids = [] # (V_unclustered) list containing IDs in output (clustered) cond
    # so non_clustered_id[instance_idx] = cluster_idx
    for i, (size, part, port, macro, _) in enumerate(components):
        if port or macro:
            non_clustered_ids.append(len(data_macros))
            data_x.append(size)
            data_ports.append(port)
            data_macros.append(macro)
        else:
            non_clustered_ids.append(part)
            obj_area = size[0]*size[1]
            cluster_area[part] += obj_area
            cluster_centroid[:, part, :] += obj_area * placements[:, i, :]
    macro_port_placements = placements[:, torch.logical_or(input_cond.is_macros, input_cond.is_ports), :]
    cluster_placements = cluster_centroid / torch.stack(cluster_area, dim=0).view((1, num_clusters, 1)).to(device=cluster_centroid.device)
    output_placements = torch.cat((cluster_placements, macro_port_placements), dim = 1)

    # ensuring a 1:1 aspect ratio
    data_x_macro = [torch.tensor([math.sqrt(cluster_area[part]), math.sqrt(cluster_area[part])], dtype=input_cond.x.dtype) for part in range(num_clusters)]
    data_x = data_x_macro + data_x

    # new edges should specify the index of the cluster
    new_id_raw = lambda u : components[u][1] if not(components[u][2] or components[u][3]) else non_clustered_ids[u]
    new_id = lambda x : new_id_raw(x.item())

    # generate new pin ids
    if "edge_pin_id" in input_cond:
        edge_index_unique = input_cond.edge_index[:, :E//2].T # (E, 2)
        edge_pin_id_unique = input_cond.edge_pin_id[:E//2, :] # (E, 2)
        sources = torch.cat((
            edge_index_unique[:,0:1].double(), 
            edge_pin_id_unique[:,0:1].double(),
            ), dim=1)
        dests = torch.cat((
            edge_index_unique[:,1:2].double(),
            edge_pin_id_unique[:,1:2].double(),
            ), dim=1)
        edge_endpoints = torch.cat((sources, dests), dim=0) # (2E, 2)
        _, global_pin_ids = torch.unique(edge_endpoints, return_inverse=True, dim=0) # (E_u, 3), (2E)
        global_pin_ids = global_pin_ids.view(2, E//2)
        reverse_pin_ids = torch.cat((global_pin_ids[1:2, :], global_pin_ids[0:1, :]), dim=0)
        global_pin_ids = torch.cat((global_pin_ids, reverse_pin_ids), dim=1) # (2, 2E)
        output_edge_pin_id = torch.stack([
            torch.tensor([u_id, v_id], dtype=input_cond.edge_pin_id.dtype)
            for (u, v), (u_id, v_id) in zip(input_cond.edge_index.T, global_pin_ids.T)
            if new_id(u) != new_id(v)])

    # ensuring all ports are at the center of the macro
    # port_macro = [data_x_macro[part][0]/2 for part in range(num_clusters)]
    new_port = lambda u, e: 0 if u < num_clusters else e
    try: # DEBUGGING TODO
        output_edge_index = torch.stack([
            torch.tensor([new_id(u), new_id(v)], dtype=input_cond.edge_index.dtype) 
            for (u, v), e in zip(input_cond.edge_index.T, input_cond.edge_attr)
            if new_id(u) != new_id(v)]).movedim(-1, 0)
        output_edge_attr = torch.stack([
            torch.tensor([new_port(new_id(u), e[0]), new_port(new_id(u), e[1]), new_port(new_id(v), e[2]), new_port(new_id(v), e[3])], dtype=input_cond.edge_attr.dtype)
            for (u, v), e in zip(input_cond.edge_index.T, input_cond.edge_attr)
            if new_id(u) != new_id(v)])
        output_cluster_map = torch.tensor(non_clustered_ids, dtype = torch.int)
    except:
        import ipdb; ipdb.set_trace()

    if (output_placements.isinf().any().item() or output_placements.isnan().any().item()):
        empty_clusters = set(assigned_parts) - set(non_clustered_ids)
        zero_area_clusters = set([i for i, area in enumerate(cluster_area) if area.item() <= 1e-12])
        print("WARNING: Empty cluster(s) detected: ", input_cond, empty_clusters, zero_area_clusters)
        # deal with clusters with no standard cells
        for empty_cluster_idx in sorted(empty_clusters, reverse=True):
            # remove entry for empty cluster from masks and placement
            data_x.pop(empty_cluster_idx)
            data_ports.pop(empty_cluster_idx)
            data_macros.pop(empty_cluster_idx)
            output_placements = torch.cat((output_placements[:, :empty_cluster_idx, :], output_placements[:, (empty_cluster_idx+1):, :]), dim=1)
            # remap edge_index and cluster_map
            output_edge_index = output_edge_index - (output_edge_index > empty_cluster_idx).int()
            output_cluster_map = output_cluster_map - (output_cluster_map > empty_cluster_idx).int()
        assert not (output_placements.isinf().any().item() or output_placements.isnan().any().item()), "nans still present in data"
        assert output_edge_index.max() < len(data_x), "edge indices mapped incorrectly while removing empty clusters"
        assert output_cluster_map.max() < len(data_x), "output cluster maps generated incorrectly while removing empty clusters"

    output_x = torch.stack(data_x, dim=0)
    output_ports = torch.stack(data_ports, dim=0)
    output_macros = torch.stack(data_macros, dim=0)

    output_cond = Data(
        x = output_x,
        is_ports = output_ports,
        is_macros = output_macros,
        edge_index = output_edge_index,
        edge_attr = output_edge_attr,
        cluster_map = output_cluster_map,
    )
    if "edge_pin_id" in input_cond:
        output_cond.edge_pin_id = output_edge_pin_id
    # (shallow) copy over other attributes in cond
    for k in input_cond.keys():
        if not k in output_cond:
            output_cond[k] = input_cond[k]

    # return to original device
    placements.to(device = placements_device)
    input_cond.to(device = cond_device)

    return output_cond.to(device = cond_device), output_placements.to(device = placements_device)

def uncluster(clustered_cond, clustered_x):
    # generate original placements based on clustered cond and placements
    # clustered_cond contains cluster_map, which is a (V_unclustered) tensor
    # map: unclustered_id -> clustered_id
    # clustered_x: (B, V, 2)
    # returns x (B, V_unclustered, 2)
    unclustered_x = clustered_x[:, clustered_cond.cluster_map, :]
    return unclustered_x

def preprocess_graph(x, cond, chip_size = (1, 1), scale = 1, chip_offset = (0, 0)):
    # x: numpy float64 array with shape (V, 2) describing 2D position on canvas
    # cond.x: torch float32 tensor (V, 2) describing instance sizes
    # cond.edge_attr: torch float64 tensor (E, 4)
    # chip_size: tuple of length 2; size of canvas in um
    # chip_size: tuple of length 2; bottom left corner coordinates in um
    # NOTE This function applies changes in-place
    if "chip_size" in cond: 
        if len(cond.chip_size) == 4: # chip_size is [x_start, y_start, x_end, y_end]
            chip_size = (cond.chip_size[2] - cond.chip_size[0], cond.chip_size[3] - cond.chip_size[1])
            chip_offset = (cond.chip_size[0], cond.chip_size[1])
        else:
            chip_size = (cond.chip_size[0], cond.chip_size[1])
            chip_offset = (0, 0)
    chip_size = torch.tensor(chip_size, dtype = torch.float32).view(1, 2)
    chip_offset = torch.tensor(chip_offset, dtype = torch.float32).view(1, 2)

    # normalizes input data
    cond.x = 2 * (cond.x / chip_size)
    cond.edge_attr = cond.edge_attr.float() 
    
    # scale edge_attr with canvas size
    cond.edge_attr[:,:2] = 2 * (cond.edge_attr[:,:2] / chip_size)
    cond.edge_attr[:,2:4] = 2 * (cond.edge_attr[:,2:4] / chip_size)

    # normalize placement data TODO fix torch.tensor warnings
    x = (torch.tensor(x, dtype=torch.float32) - chip_offset)/scale
    x = 2 * (x / chip_size) - 1
    
    # use center of instance as coordinate point and reference for terminal
    x = x + cond.x/2
    u_shape = cond.x[cond.edge_index[0,:]]
    v_shape = cond.x[cond.edge_index[1,:]]
    cond.edge_attr[:,:2] = cond.edge_attr[:,:2] - u_shape/2
    cond.edge_attr[:,2:4] = cond.edge_attr[:,2:4] - v_shape/2
    return x, cond

def postprocess_placement(x, cond, chip_size=None, process_graph=False):
    """
    Assumes x is (V, 2), placement is 2D coordinates, with no rotations
    chip_size must be tensor
    Note: if process_graph=True, postprocessing is done in-place
    """
    if chip_size is None:
        if "chip_size" in cond:
            cond_chip_size = torch.tensor(cond.chip_size, dtype = torch.float32) if not isinstance(cond.chip_size, torch.Tensor) else cond.chip_size
            if len(cond.chip_size) == 2:
                chip_size = cond_chip_size
                chip_offset = torch.zeros_like(chip_size)
            elif len(cond.chip_size) == 4:
                chip_size = cond_chip_size[2:] - cond_chip_size[:2]
                chip_offset = cond_chip_size[:2]
        else:
            return x # no normalization to be done
    else:
        chip_size = chip_size if isinstance(chip_size, torch.Tensor) else torch.tensor(chip_size)
    scale = chip_size.view(1, 2).to(device = x.device)
    chip_offset = chip_offset.to(device = x.device)
    
    x = x - cond.x/2
    x = scale * (x+1)/2
    x = x + chip_offset

    if not process_graph:
        return x
    else:
        # use bottom left as reference for terminal coordinates
        u_shape = cond.x[cond.edge_index[0,:]]
        v_shape = cond.x[cond.edge_index[1,:]]
        cond.edge_attr[:,:2] = cond.edge_attr[:,:2] + u_shape/2
        cond.edge_attr[:,2:4] = cond.edge_attr[:,2:4] + v_shape/2

        # un-normalize terminal coordinates
        cond.edge_attr[:,:2] = (cond.edge_attr[:,:2] * scale)/2
        cond.edge_attr[:,2:4] = (cond.edge_attr[:,2:4] * scale)/2

        # un-normalize object sizes
        cond.x = (cond.x * scale)/2
        return x, cond

def get_pickle_paths(output_dir, idx):
    graph_path = os.path.join(output_dir, f'graph{idx}.pickle')
    placement_path = os.path.join(output_dir, f'output{idx}.pickle')
    return graph_path, placement_path

def write_to_pickle(x, cond, output_dir, idx):
    """
    Writes np array x and cond Data object to output_dir.
    Note that idx should be 0-indexed
    """
    graph_path, placement_path = get_pickle_paths(output_dir, idx)
    print(f"pickling to {graph_path} and {placement_path}")
    with open(graph_path, 'wb') as fp:
        pickle.dump(cond.to(device="cpu"), fp)
    with open(placement_path, 'wb') as fp:
        pickle.dump(x.to(device="cpu"), fp)

def open_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def dict_to_csv(out_dict, filename):
    """
    Dict has key: List[data1, data2, ...]
    """
    with open(filename, 'w') as csv_file:  
        writer = csv.writer(csv_file,delimiter=',')
        out_columns = [[k] + v for k, v in out_dict.items()]
        for row in zip(*out_columns):
            writer.writerow(row)
    return

def visualize_placement(x, cond, plot_pins = False, plot_edges = False, img_size = (256, 256), mask = None):
    """ 
    Visualizes the X with node attributes, returning an numpy image
    All coordinates are normalized w.r.t canvas size
    x is (V, 2) tensor with 2D coordinates describing placement of center of instances
    cond is pytorch geometric Data object with the following:
    - x is (V, 2) tensor with sizes of instances
    - edge_index (2, E)
    - edge_attr (E, 4) tensor describing pin locations, measured relative to center of instance
    - mask is mask override
    """
    width, height = img_size
    background_color = "white"
    base_image = Image.new("RGBA", (width, height), background_color)

    assert len(x.shape) == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    assert x.shape[1] == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    x = x[:,:2]

    def canvas_to_pixel_coord(x):
        # x is (B, 2) tensor representing normalized 2D coordinates in canvas space
        output = torch.zeros_like(x)
        output[:,0] = (0.5 + x[:,0]/2) * width
        output[:,1] = (0.5 - x[:,1]/2) * height
        return output

    V, _ = x.shape
    mask = cond.is_ports if "is_ports" in cond and mask is None else mask
    h_step = 0.2 / V if "is_macros" in cond else 1.0 / V
    h_offsets = {"macro": .0, "port": 0.35, "sc": 0.55}

    left_bottom = x - cond.x/2
    right_top = x + cond.x/2
    inbounds = torch.logical_and(left_bottom >= -1, right_top <= 1)
    inbounds = torch.logical_and(inbounds[:,0], inbounds[:,1])

    left_bottom_px = canvas_to_pixel_coord(left_bottom)
    right_top_px = canvas_to_pixel_coord(right_top)
    
    for i in range(V):
        image = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        if "is_macros" in cond:
            if cond.is_macros[i]:
                h_offset = h_offsets["macro"] 
            elif mask is None or not mask[i]:
                h_offset = h_offsets["sc"]
            else:
                h_offset = h_offsets["port"]
        else:
            h_offset = 0.0
        color = hsv_to_rgb(
            i * h_step + h_offset, 
            1 if (mask is None or not mask[i]) else 0.2, 
            0.9 if inbounds[i] else 0.5,
        )
        draw.rectangle([left_bottom_px[i,0], right_top_px[i,1], right_top_px[i,0], left_bottom_px[i,1]], fill=(*color, 160), width=0)
        base_image = Image.alpha_composite(base_image, image)

    draw = ImageDraw.Draw(base_image)
    # get pin positions
    if plot_edges or plot_pins:
        unique_edges = cond.edge_attr.shape[0]//2
        u_pos = cond.edge_attr[:unique_edges,:2] + x[cond.edge_index[0,:unique_edges]]
        v_pos = cond.edge_attr[:unique_edges,2:4] + x[cond.edge_index[1,:unique_edges]]
        u_pos = canvas_to_pixel_coord(u_pos)
        v_pos = canvas_to_pixel_coord(v_pos)

    # plot edges
    if plot_edges:
        for i in range(unique_edges):
            draw.line([tuple(u_pos[i].detach().cpu().numpy()), tuple(v_pos[i].detach().cpu().numpy())], fill="gray")
    # plot pin positions
    if plot_pins:
        draw.point([(row[0], row[1]) for row in u_pos.detach().cpu().numpy()], fill="black")
        draw.point([(row[0], row[1]) for row in v_pos.detach().cpu().numpy()], fill="yellow")
    
    return np.array(base_image)[:,:,:3]

def hsv_to_rgb(h, s, v):
    """
    Converts HSV (Hue, Saturation, Value) color space to RGB (Red, Green, Blue).
    h: float [0, 1] - Hue
    s: float [0, 1] - Saturation
    v: float [0, 1] - Value
    Returns: tuple (r, g, b) representing RGB values in the range [0, 255]
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)

def debug_plot_img(x, name = "debug_img", rescale = False, autoscale = False):
    # x is (C, H, W) image, this function plots and saves to file
    # assumes images are [-1, 1]
    import matplotlib.pyplot as plt
    # scaling
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if len(x.shape)==3 and (x.shape[-1] not in [1, 2, 3]):
        x = x.moveaxis(0,-1)
    if rescale:
        x = (x + 1)/2 if not autoscale else (x-x.min())/(x.max()-x.min())
    plt.figure()
    plt.imshow(x)
    plt.savefig(name, dpi=1000)

def debug_plot_graph(x, name = "debug_img", fig_title = None):
    # x is (D) vector, this function plots and saves to file
    # assumes images are [-1, 1]
    import matplotlib.pyplot as plt
    # scaling
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    plt.figure()
    plt.plot(x)
    if fig_title is not None:
        plt.title(fig_title)
    plt.savefig(name, dpi=1000)