import torch
import torch_geometric.utils as tgu
import guidance
import matplotlib.pyplot as plt
import numpy as np
import utils
import scipy.sparse.linalg as linalg
import os

def get_num_vertices(x, cond):
    return x.shape[0]

def get_num_edges(x, cond):
    return cond.edge_index.shape[1]

def get_degree(x, cond):
    # out degrees in first half, in deg in second
    degrees = tgu.degree(cond.edge_index[0], num_nodes = x.shape[0]).int() 
    return degrees

def get_degree_split(x, cond):
    """
    returns vertex degrees for macros, ports, and SCs individually (in that order)
    """
    macro_mask, port_mask, sc_mask = get_masks(x, cond)
    degrees = get_degree(x, cond) # (V,)
    macro_degrees = degrees[macro_mask]
    port_degrees = degrees[port_mask]
    sc_degrees = degrees[sc_mask]
    return macro_degrees, port_degrees, sc_degrees

def get_num_neighbors(x, cond):
    unique_edges = torch.unique(cond.edge_index.T, return_inverse=False, dim=0)
    neighbors = tgu.degree(unique_edges.T[0], num_nodes = x.shape[0]).int()
    return neighbors

def get_neighbors_split(x, cond):
    """
    returns neighbors for macros, ports, and SCs individually (in that order)
    """
    macro_mask, port_mask, sc_mask = get_masks(x, cond)
    neighbors = get_num_neighbors(x, cond) # (V,)
    macro_neighbors= neighbors[macro_mask]
    port_neighbors = neighbors[port_mask]
    sc_neighbors = neighbors[sc_mask]
    return macro_neighbors, port_neighbors, sc_neighbors

def get_net_hpwl(x, cond):
    hpwl_net = guidance.HPWL()
    pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond)
    hpwl_net = hpwl_net(x, pin_map, pin_offsets, pin_edge_index, net_aggr="none")
    return hpwl_net

def get_masks(x, cond):
    """
    returns masks for macros, ports, and SCs (in that order) that are mutually exclusive
    """
    V = cond.num_nodes
    macro_mask = cond.is_macros if "is_macros" in cond else torch.zeros((V,), dtype=bool, device=cond.x.device)
    port_mask = cond.is_ports if "is_ports" in cond else torch.zeros((V,), dtype=bool, device=cond.x.device)
    assert (macro_mask & port_mask).sum().item() == 0, "macros and ports should be mutually exclusive"
    
    sc_mask = ~(port_mask | macro_mask)
    return macro_mask, port_mask, sc_mask

def get_areas(x, cond):
    macro_mask, port_mask, sc_mask = get_masks(x, cond)
    macro_sizes = cond.x[macro_mask, :]
    port_sizes = cond.x[port_mask, :]
    sc_sizes = cond.x[sc_mask, :]

    macro_areas = macro_sizes[:, 0] * macro_sizes[:, 1]
    port_areas = port_sizes[:, 0] * port_sizes[:, 1]
    sc_areas = sc_sizes[:, 0] * sc_sizes[:, 1]
    return macro_areas, port_areas, sc_areas

def get_density(x, cond):
    areas = cond.x[:, 0] * cond.x[:, 1]
    density = areas.sum() / 4.0
    return density.item()

def get_spectral_info(x, cond, k = 10, normalization = 'rw'):
    """
    Obtain important spectral information
    - normalization can be None, 'sym', or 'rw'
    """
    edge_index, edge_weight = tgu.get_laplacian(cond.edge_index, None,
                                                normalization=normalization,
                                                num_nodes=cond.num_nodes)

    L = tgu.to_scipy_sparse_matrix(edge_index, edge_weight, cond.num_nodes)

    try:
        eig_vals, eig_vecs = linalg.eigs(L, k=k+1, which='SM', return_eigenvectors=True)
    except:
        return np.array([-1] * (k+1), dtype=np.float32) # dummy value, since eigenvalue >= 0 nominally
    # sometimes eig values and vectors are not sorteds
    argsort_indices = eig_vals.real.argsort()
    eig_vals = eig_vals.real[argsort_indices]
    eig_vecs = eig_vecs.real[:, argsort_indices]
    return eig_vals[1:] # first eigenvalue should be 0

def get_average_edge_length(x, cond):
    edge_length = utils.edge_length(x, cond)
    num_edges = cond.edge_index.shape[1]
    return edge_length/num_edges

def get_edge_splits(x, cond):
    """
    Get # edges between macros and SCs
    WARNING we assume masks are mutually exclusive
    returns a dict{key: int} containing number of edges
    """
    E = cond.num_edges//2
    unique_edges = cond.edge_index[:, :E] # (2, E)

    select_indices = [0, 2]
    select_names = ["macro", "sc"] 
    masks = get_masks(x, cond)

    type_label = torch.zeros_like(masks[0]) # (V,)
    for i in select_indices:
        type_label = type_label + i * masks[i].int() 
    unique_edge_labels = type_label[unique_edges]

    output = {}
    for name_src, index_src in zip(select_names, select_indices):
        match_src = (unique_edge_labels[0, :] == index_src)
        for name_dest, index_dest in zip(select_names, select_indices):
            match_dest = (unique_edge_labels[1, :] == index_dest)
            match_both = match_src & match_dest
            output[f"{name_src}_to_{name_dest}_edges"] = match_both.sum().item()
    return output

def get_edge_densities(unique_edges, edge_splits):
    output = {}
    for name, split in edge_splits.items():
        new_name = "_".join(name.split("_")[:-1] + ['density'])
        output[new_name] = split/unique_edges
    return output

def analyze_sample(x, cond):
    V = get_num_vertices(x, cond)
    E = get_num_edges(x, cond)
    hpwl_net = get_net_hpwl(x, cond)
    hpwl_net = hpwl_net[hpwl_net.nonzero(as_tuple=True)]

    hpwl = hpwl_net.sum(dim=-1)
    num_nets = hpwl_net.shape[-1]
    # hpwl_slow = utils.hpwl(x, cond)

    degree = get_degree(x, cond)
    neighbors = get_num_neighbors(x, cond)
    density = get_density(x, cond)

    # spectral analysis
    eig_vals = get_spectral_info(x, cond, k=1)

    average_edge_length = get_average_edge_length(x, cond)

    # decompose into macros, cells, ports
    macro_mask, port_mask, sc_mask = get_masks(x, cond)
    num_macro = macro_mask.sum().item()
    num_port = port_mask.sum().item()
    num_sc = sc_mask.sum().item()

    macro_areas, port_areas, sc_areas = get_areas(x, cond)
    macro_degrees, port_degrees, sc_degrees = get_degree_split(x, cond)
    macro_neighbors, port_neighbors, sc_neighbors = get_neighbors_split(x, cond)

    edge_splits = get_edge_splits(x, cond)
    edge_densities = get_edge_densities(E//2, edge_splits)

    metrics = {
        "num_vertices": V,
        "num_edges": E,
        "density": density,
        "hpwl": hpwl.cpu(),
        "num_nets": num_nets,
        "net_hpwl": hpwl_net.cpu(),
        "degree": degree.cpu(),
        "neighbors": neighbors.cpu(),
        "mean_edge_length": average_edge_length.cpu(),
        "lambda_2": eig_vals[0],
        "num_macro": num_macro,
        "num_port": num_port,
        "num_sc": num_sc,
        "macro_areas": macro_areas.cpu(),
        "port_areas": port_areas.cpu(),
        "sc_areas": sc_areas.cpu(),
        "macro_degrees": macro_degrees.cpu(),
        "port_degrees": port_degrees.cpu(),
        "sc_degrees": sc_degrees.cpu(),
        "macro_neighbors": macro_neighbors.cpu(),
        "port_neighbors": port_neighbors.cpu(),
        "sc_neighbors": sc_neighbors.cpu(),
        **edge_splits,
        **edge_densities,
    }
    metrics_special = {
        "idx": cond.file_idx if "file_idx" in cond else 0,
        "num_vertices": V,
        "num_edges": E,
        "density": density,
        "mean_degree": degree.float().mean().cpu(),
        "mean_num_neighbors": neighbors.float().mean().cpu(),
        "num_nets": num_nets,
        "mean_edge_length": average_edge_length.cpu(),
        "lambda_2": eig_vals[0],
        "hpwl": hpwl.cpu(),
        "mean_net_hpwl": hpwl_net.mean().cpu(),
        "num_macro": num_macro,
        "num_port": num_port,
        "num_sc": num_sc,
        "mean_macro_degrees": macro_degrees.float().mean().cpu(),
        "mean_port_degrees": port_degrees.float().mean().cpu(),
        "mean_sc_degrees": sc_degrees.float().mean().cpu(),
        "mean_macro_neighbors": macro_neighbors.float().mean().cpu(),
        "mean_port_neighbors": port_neighbors.float().mean().cpu(),
        "mean_sc_neighbors": sc_neighbors.float().mean().cpu(),
        **edge_splits,
        **edge_densities,
    }
    return metrics, metrics_special
    

def generate_histograms(collections, log_dir, bins=100, save_txt=False):
    # TODO generate compound histograms

    # generate individual histograms
    for k, v in collections.items():
        if np.issubdtype(v.dtype, np.integer) and v.size > 0:
            v_range = v.max() - v.min()
            v_min = v.min()
            bin_size = max(v_range//bins, 1) # use bins as minimum number of bins; max is 2x the minimum
            num_bins = max(np.ceil(v_range/bin_size).astype(int), 1)
            density, edges = np.histogram(v, bins=num_bins, range=(v_min, v_min+bin_size*num_bins), density=False)
        else:
            density, edges = np.histogram(v, bins=bins, density=True)
        fig, ax = plt.subplots(1, 1)
        ax.stairs(density, edges)
        fig.savefig(os.path.join(log_dir, f"{k}__hist.png"), dpi=300)
        plt.close(fig)
        if save_txt:
            np.savetxt(os.path.join(log_dir, f"{k}__density.csv"), density, delimiter=',')
            np.savetxt(os.path.join(log_dir, f"{k}__edges.csv"), edges, delimiter=',')

def generate_scatterplots(collections, scatter_keys, log_dir=None, logger=None):
    for plot_keys in scatter_keys:
        x_name = plot_keys[0]
        y_name = plot_keys[1]
        if x_name in collections and y_name in collections:
            plot_name = f"{x_name}_vs_{y_name}"
            if logger is not None:
                scatter_plot = utils.plot_scatter(collections[x_name], collections[y_name], x_title=x_name, y_title=y_name)
                logger.add({plot_name: scatter_plot})
            if log_dir is not None:
                fig, ax = plt.subplots(1, 1)
                ax.scatter(collections[x_name], collections[y_name])
                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)
                fig.savefig(os.path.join(log_dir, f"{plot_name}__hist.png"), dpi=300)