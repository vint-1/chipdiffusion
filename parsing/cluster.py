import parse_utils
import utils
import torch
import os
import pathlib
from collections import OrderedDict
import hydra
import re
from omegaconf import OmegaConf, open_dict

class Cluster:
    """
    Object for representing standard cell cluster
    """
    def __init__(self, position, shape, scale_factor):
        """
        - position: np array (2,) containing (x, y) positions of object (bottom left corner)
        - shape: np array (2,) containing (x, y) shape of object
        TODO figure out scaling factor
        """
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        if isinstance(shape, torch.Tensor):
            shape = shape.cpu().numpy()
        self.scale_factor = scale_factor
        self.position = position
        self.shape = shape
        self.pins = OrderedDict() # note values might have 0-length
        self.pin_ids = None # tuple(obj, pin) -> pin_name: str
    
    def add_component(self, component):
        """
        component is DesignComponent object
        """
        assert component.name not in self.pins, "component should only be added to cluster once"
        self.pins[component.name] = list(component.pins.keys())
        self.pin_ids = None

    def remove_pin(self, obj_name, pin):
        """
        remove specified pin. Throws AssertionError if obj and pin do not exist
        - obj_name: str
        - pin: pin name as str
        """
        self.pins[obj_name].remove(pin)
        self.pin_ids = None
    
    def __repr__(self):
        return f"Cluster(pos={self.position}, shape={self.shape}, components={list(self.pins.keys())})"

    def _create_pin_ids(self):
        """
        Generate names for pins in cluster
        """
        idx = 1 # pin numbers traditionally 1-indexed
        self.pin_ids = {}
        for component, component_pins in self.pins.items():
            for component_pin in component_pins:
                self.pin_ids[(component, component_pin)] = f"P{idx}"
                idx += 1

    def get_pin_id(self, pin):
        """
        - pin: tuple(unclustered_name, pin_name)
        """
        if self.pin_ids is None:
            self._create_pin_ids()
        return self.pin_ids[pin]

def cluster_lefdef(lef_path, def_path, out_dir, pickle_out_dir=None, pickle_idx=None, num_clusters=512, cluster_ubfactor=5, ref_placement_path = None, verbose=False, mode="remove_first", cluster_algorithm="hmetis"):
    """
    Perform clustering for LEF/DEF format and output into specified folder
    """
    if mode == "remove_first":
        cluster_fn = utils.cluster_2
    elif mode == "remove_after":
        cluster_fn = utils.cluster
    else:
        raise NotImplementedError
    if num_clusters == 0:
        def no_cluster(data_obj, num_clusters, placements = None, **kwargs):
            assert placements is not None, "placements need to be provided for no-op clustering"
            return data_obj, placements
        cluster_fn = no_cluster
    
    # filter out -NNN.gp.def from def paths if reference placement is used
    def_out_name = re.sub("^(\S+)-([0-9]+).gp.def$","\\1.def", pathlib.PurePath(def_path).name)
    def_out_path = os.path.join(out_dir, def_out_name)
    lef_out_path = os.path.join(out_dir, pathlib.PurePath(lef_path).name)
    is_skip = (num_clusters == 0) or (os.path.exists(def_out_path) and os.path.exists(lef_out_path))
    if pickle_out_dir is not None:
        pickle_graph_path, pickle_placement_path = utils.get_pickle_paths(pickle_out_dir, pickle_idx)
        is_skip = is_skip and os.path.exists(pickle_graph_path) and os.path.exists(pickle_placement_path)
    if is_skip:
        print(f"Output LEF/DEF at {out_dir} and pickles at {pickle_out_dir} already exist for {def_path.name}.")
        # load pickle and generate metrics
        # TODO load original graph too to get original vertices and edges
        metrics = {}
        if pickle_out_dir is not None:
            clustered_graph = utils.open_pickle(pickle_graph_path)
            metrics = {
                "name": re.search("(.*).lef$",lef_path.name).group(1),
                "num_vertices": clustered_graph.x.shape[0],
                "num_edges": clustered_graph.edge_index.shape[1],
                "num_macros": clustered_graph.is_macros.sum().item(),
                "num_ports": clustered_graph.is_ports.sum().item(),
            }
        return metrics
    
    lib_components, pins = parse_utils.parse_lef_file(lef_path)
    if ref_placement_path is not None:
        placement_components, _, _, _, _ = parse_utils.parse_def_file(ref_placement_path, only_netlist=False)
        ref_multiplier = parse_utils.DesignComponent.mult
    components, nets_graph, nets_list, nets_routing, chip_size = parse_utils.parse_def_file(def_path, only_netlist=False)
    scale_factor = parse_utils.DesignComponent.mult # I hate this global variable nonsense
    if ref_placement_path is not None:
        parse_utils.copy_ref_placement(placement_components, components, placement_mult=ref_multiplier, target_mult=scale_factor)
    
    # add pin data from LEF to graph
    parse_utils.update_design_components(components, lib_components, pins)
    data_obj = parse_utils.to_torch_data(components, nets_graph, "N", chip_size)

    # cluster standard cells (technically we should preprocess, then postprocess to get the right cluster locations?)
    placement = torch.tensor(parse_utils.get_locations(components, chip_size, "N"))

    placement, data_obj = utils.preprocess_graph(placement, data_obj)
    debug_plot(placement, data_obj, f"logs/temp/debug_{pathlib.Path(def_path).name}.png")
    clustered_graph, clustered_placement = cluster_fn(data_obj, num_clusters, ubfactor = cluster_ubfactor, temp_dir = "logs/temp", verbose = verbose, placements = placement.unsqueeze(dim=0), algorithm = cluster_algorithm)
    debug_plot(clustered_placement.squeeze(dim=0), clustered_graph, f"logs/temp/debug_clustered_{pathlib.Path(def_path).name}.png")
    clustered_placement, clustered_graph = utils.postprocess_placement(clustered_placement, clustered_graph, process_graph=True)

    # write clustered graphs to pickle if needed
    if pickle_out_dir is not None:
        assert isinstance(pickle_idx, int), "pickle_idx must be specified if writing to pickle."
        utils.write_to_pickle(clustered_placement.squeeze(dim=0), clustered_graph, pickle_out_dir, pickle_idx)
    
    metrics = {
        "name": re.search("(.*).lef$",lef_path.name).group(1),
        "num_vertices": clustered_graph.x.shape[0],
        "num_edges": clustered_graph.edge_index.shape[1],
        "num_macros": clustered_graph.is_macros.sum().item(),
        "num_ports": clustered_graph.is_ports.sum().item(),
        "original_vertices": data_obj.x.shape[0],
        "original_edges": data_obj.edge_index.shape[1],
    }

    if num_clusters == 0:
        return metrics

    # Preparing for outputs
    # Generate table of cluster objects with placement and shapes
    is_cluster = ~clustered_graph.is_ports & ~clustered_graph.is_macros # bool(V_clustered,)
    # clustered_component_idx -> Cluster
    clusters = {i: Cluster(clustered_placement[0,i,:], clustered_graph.x[i], scale_factor) for i, x in enumerate(is_cluster) if x} 

    # Add pins to clusters
    clustered_id_map = {} # obj_name -> clustered_component_idx
    for i, (name, component) in enumerate(components.items()):
        clustered_idx = clustered_graph.cluster_map[i].item()
        clustered_id_map[name] = clustered_idx
        if bool(is_cluster[clustered_idx]): # component belongs to a cluster
            clusters[clustered_idx].add_component(component)

    # Iterate through netlist and remove internal edges
    # ie. if source cluster == dest cluster, remove dest from netlist
    # then remove all netlists with <= 1 pin
    culled_nets_list = []
    for (src_obj, src_pin), dests in nets_list:
        src_cluster = clustered_id_map[src_obj]
        if src_cluster in clusters: # src is actually a standard cell cluster
            external_dests = []
            for dest_obj, dest_pin in dests:
                if src_cluster != clustered_id_map[dest_obj]:
                    external_dests.append((dest_obj, dest_pin))
        else:
            external_dests = dests
        if len(external_dests) > 0:
            culled_nets_list.append(((src_obj, src_pin), external_dests))

    # Remove culled pins from cluster objects
    all_pins = get_all_pins(nets_list)
    remaining_pins = get_all_pins(culled_nets_list)
    culled_pins = all_pins - remaining_pins
    for (obj, pin) in culled_pins:
        cluster = clusters[clustered_id_map[obj]]
        cluster.remove_pin(obj, pin)

    # Writing in DEF/LEF format
    base_def = parse_utils.DefFile(def_path)
    base_def.update_components(clustered_id_map, clusters, culled_nets_list)
    base_def.write_output(def_out_path)
    print(f"Writing DEF output to {def_out_path}")

    # Write LEF outputs
    base_lef = parse_utils.LefFile(lef_path)
    component_map = {k: v.type for k,v in components.items()}
    base_lef.update_components(component_map, clustered_id_map, clusters, culled_nets_list)
    base_lef.write_output(lef_out_path)
    print(f"Writing LEF output to {lef_out_path}")

    return metrics

def get_all_pins(nets_list):
    all_pins = set()
    for src, dest in nets_list:
        all_pins.add(src)
        all_pins.update(dest)
    return all_pins

def debug_plot(placement, data, filename = "logs/temp/debug_img.png"):
    img = utils.visualize_placement(placement, data, plot_pins = True, plot_edges = False, img_size=(1024,1024))
    utils.debug_plot_img(img, filename)

@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg):
    benchmark_name = pathlib.PurePath(cfg.benchmark_dir).name

    out_name = f"{benchmark_name}.cluster{cfg.num_clusters}.{cfg.name}"
    out_dir = os.path.join(cfg.out_dir, out_name)
    pickle_out_dir = os.path.join(cfg.pickle_out_dir, out_name)
    os.makedirs(pickle_out_dir, exist_ok=True)
    num_clusters = cfg.num_clusters # NOTE that number of clusters cannot be more than vertices in graph
    
    if cfg.use_placed_def:
        def_files = sorted(list(pathlib.Path(cfg.benchmark_dir).rglob("*_placed.def"))) # *_placed.def
        lef_pattern = "^(\S+)_placed.def"
    else:
        def_files = sorted(list(pathlib.Path(cfg.benchmark_dir).rglob("*.def")))
        lef_pattern = "^(\S+).def"

    print(f"==== Clustering into {num_clusters} clusters, output dir: {out_name} ====")
    clustered_metrics = {}
    for i, def_path in enumerate(def_files):
        circuit_dir = def_path.parent
        def_name = def_path.name
        lef_name = re.sub(lef_pattern,"\\1.lef", def_name)
        pickle_idx = int(re.search("([0-9]+)", def_name).group(1)) - 1 # pickles are 0-indexed
        
        lef_paths = list(pathlib.Path(circuit_dir).glob(lef_name))
        if len(lef_paths)!=1:
            print(f"WARNING: {len(lef_paths)} matching LEF files found for {def_path}. skipping...")
            continue
        lef_path = lef_paths[0] # raises an error if appropriate lef not found

        # find reference placement if needed (for oracle clustering)
        if cfg.placement_dir is not None:
            search_pattern = re.sub(lef_pattern, "\\1*.def", def_name)
            placement_def = sorted(list(pathlib.Path(cfg.placement_dir).rglob(search_pattern)))
            placement_path = placement_def[0] # select first path lexicographically
        else:
            placement_path = None

        circuit_out_dir = os.path.join(out_dir, circuit_dir.name)
        os.makedirs(circuit_out_dir, exist_ok=True)
        print(f"Starting clustering on {circuit_dir.name}.  LEF: {lef_path}  DEF: {def_path}")
        circuit_metrics = cluster_lefdef(
            lef_path, 
            def_path, 
            circuit_out_dir, 
            pickle_out_dir=pickle_out_dir, 
            pickle_idx=pickle_idx, 
            num_clusters=num_clusters,
            cluster_ubfactor=cfg.ubfactor, 
            verbose=cfg.verbose,
            mode=cfg.mode,
            cluster_algorithm=cfg.cluster_algorithm,
            ref_placement_path=placement_path,
            )
        for k, v in circuit_metrics.items():
            if k in clustered_metrics:
                clustered_metrics[k].append(v)
            else:
                clustered_metrics[k] = [v]

        if cfg.limit > 0 and (i+1) >= cfg.limit:
            break

    # Write info
    utils.dict_to_csv(clustered_metrics, os.path.join(out_dir, "info.csv"))

if __name__ == "__main__":
    main()