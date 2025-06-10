import torch
import math
from torch_geometric.data import Data
import sklearn.cluster

def kmeans_cluster(cond, num_clusters, placements):
    """
    placements should be (1, V, 2) tensor
    Note: outputs will be on same device as inputs
    """
    B, V, _ = placements.shape
    assert B == 1, "kmeans clustering cannot handle batch size > 1"
    placements = placements[0, :, :]
    weights = (~(cond.is_macros | cond.is_ports)).float()
    kmeans = sklearn.cluster.KMeans(num_clusters, init="k-means++", n_init="auto").fit(placements, sample_weight=weights)
    cluster_labels = kmeans.labels_
    return cluster_labels

def random_cluster(cond, num_clusters, placements):
    """
    placements should be (1, V, 2) tensor
    Note: outputs will be on same device as placements
    """
    B, V, _ = placements.shape
    cluster_labels = torch.randint(0, num_clusters, (V,), dtype=torch.int32, device=placements.device) # torch random
    return cluster_labels

def cluster(unclustered_cond, num_clusters, placements, algorithm = "kmeans"):
    """
    placements should be (B, V, 2) tensor.
    Note: outputs will be on same device as inputs
    """

    # move to cpu for clustering
    placements_device = placements.device
    cond_device = unclustered_cond.x.device
    placements = placements.to(device = "cpu")
    unclustered_cond.to(device = "cpu")
    _, E = unclustered_cond.edge_index.shape
    assert E % 2 == 0, "cond edge index assumed to contain forward and reverse edges"

    # perform clustering
    if algorithm == "kmeans":
        assigned_parts = kmeans_cluster(unclustered_cond, num_clusters, placements)
    elif algorithm == "random":
        assigned_parts = random_cluster(unclustered_cond, num_clusters, placements)
    else:
        raise NotImplementedError

    # create new cond_val
    B, _, _ = placements.shape
    components = list(zip(unclustered_cond.x, assigned_parts, unclustered_cond.is_ports, unclustered_cond.is_macros, range(len(unclustered_cond.is_ports))))
    data_x = []
    data_ports = [torch.tensor(False) for _ in range(num_clusters)]
    data_macros = [torch.tensor(False) for _ in range(num_clusters)]
    cluster_area = [torch.tensor(0, dtype = unclustered_cond.x.dtype) for _ in range(num_clusters)]
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
    macro_port_placements = placements[:, torch.logical_or(unclustered_cond.is_macros, unclustered_cond.is_ports), :]
    cluster_placements = cluster_centroid / torch.stack(cluster_area, dim=0).view((1, num_clusters, 1)).to(device=cluster_centroid.device)
    output_placements = torch.cat((cluster_placements, macro_port_placements), dim = 1)

    # ensuring a 1:1 aspect ratio
    data_x_macro = [torch.tensor([math.sqrt(cluster_area[part]), math.sqrt(cluster_area[part])], dtype=unclustered_cond.x.dtype) for part in range(num_clusters)]
    data_x = data_x_macro + data_x

    # new edges should specify the index of the cluster
    new_id_raw = lambda u : components[u][1] if not(components[u][2] or components[u][3]) else non_clustered_ids[u]
    new_id = lambda x : new_id_raw(x.item())

    # generate new pin ids
    if "edge_pin_id" in unclustered_cond:
        edge_index_unique = unclustered_cond.edge_index[:, :E//2].T # (E, 2)
        edge_pin_id_unique = unclustered_cond.edge_pin_id[:E//2, :] # (E, 2)
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
            torch.tensor([u_id, v_id], dtype=unclustered_cond.edge_pin_id.dtype)
            for (u, v), (u_id, v_id) in zip(unclustered_cond.edge_index.T, global_pin_ids.T)
            if new_id(u) != new_id(v)])

    # ensuring all ports are at the center of the macro
    # port_macro = [data_x_macro[part][0]/2 for part in range(num_clusters)]
    new_port = lambda u, e: 0 if u < num_clusters else e
    try: # DEBUGGING TODO
        output_edge_index = torch.stack([
            torch.tensor([new_id(u), new_id(v)], dtype=unclustered_cond.edge_index.dtype) 
            for (u, v), e in zip(unclustered_cond.edge_index.T, unclustered_cond.edge_attr)
            if new_id(u) != new_id(v)]).movedim(-1, 0)
        output_edge_attr = torch.stack([
            torch.tensor([new_port(new_id(u), e[0]), new_port(new_id(u), e[1]), new_port(new_id(v), e[2]), new_port(new_id(v), e[3])], dtype=unclustered_cond.edge_attr.dtype)
            for (u, v), e in zip(unclustered_cond.edge_index.T, unclustered_cond.edge_attr)
            if new_id(u) != new_id(v)])
        output_cluster_map = torch.tensor(non_clustered_ids, dtype = torch.int)
    except:
        import ipdb; ipdb.set_trace()

    if (output_placements.isinf().any().item() or output_placements.isnan().any().item()):
        empty_clusters = set(assigned_parts) - set(non_clustered_ids)
        zero_area_clusters = set([i for i, area in enumerate(cluster_area) if area.item() <= 1e-12])
        print("WARNING: Empty cluster(s) detected: ", unclustered_cond, empty_clusters, zero_area_clusters)
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
    if "edge_pin_id" in unclustered_cond:
        output_cond.edge_pin_id = output_edge_pin_id
    # (shallow) copy over other attributes in cond
    for k in unclustered_cond.keys():
        if not k in output_cond:
            output_cond[k] = unclustered_cond[k]

    # return to original device
    unclustered_cond.to(device = cond_device)

    return output_cond.to(device = cond_device), output_placements.to(device = placements_device)

def uncluster(clustered_cond, clustered_x):
    """
    generate original placements based on clustered cond and placements.

    Inputs:
    - clustered_cond: contains cluster_map, which is a (V_unclustered) tensor.
    cluster_map: unclustered_id -> clustered_id
    - clustered_x: (B, V, 2)

    returns: x (B, V_unclustered, 2)
    """
    unclustered_x = clustered_x[:, clustered_cond.cluster_map, :]
    return unclustered_x
