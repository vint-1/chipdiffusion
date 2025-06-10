import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import orientations
import wandb
import common
import os
import policies
import shapely
import math
import subprocess
import pathlib
from torch.utils.data import Subset
import pickle
from shapely.geometry import Polygon
from torch_geometric.data import Data
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from pathlib import Path
from collections import OrderedDict
import re
import guidance
import csv
import time
import moviepy.editor
import plotly.graph_objects as go
import matplotlib.pyplot as plt

@torch.no_grad()
def validate(x_val, model, cond=None):
    model.eval()
    t = torch.randint(1, model.max_diffusion_steps + 1, [x_val.shape[0]], device = x_val.device)
    if cond is None:
        loss, model_metrics = model.loss(x_val, t)
    else:
        loss, model_metrics = model.loss(x_val, cond, t)
    logs = {
        "loss": loss.cpu().item()
    }
    logs.update(model_metrics)
    model.train()
    return logs

@torch.no_grad()
def display_predictions(x_val, y_val, model, logger, prefix = "val", text_labels = None):
    model.eval()
    log_probs = model(x_val)
    probs = torch.nn.functional.softmax(log_probs, dim=-1)
    predictions = log_probs.argmax(dim=-1)
    for image, pred, label, prob, logit in zip(
        torch.movedim(x_val, 1, -1).cpu().numpy(), 
        predictions.cpu().numpy(), 
        y_val.cpu().numpy(),
        probs.cpu().numpy(),
        log_probs.cpu().numpy(),
        ):
        log_image = wandb.Image(image)
        logs = {
            "examples": {
                "image": log_image,
                "prediction": text_labels[pred] if text_labels else pred,
                "ground truth": text_labels[label] if text_labels else label,
                "pred prob": prob[pred],
                "truth prob": prob[label],
                "logit histogram": logit,
            },
        }
        logger.add(logs, prefix = prefix)
    model.train()

@torch.no_grad()
def display_samples(batch_size, model, logger, intermediate_every = 200, prefix = "val"):
    model.eval()
    samples, intermediates = model.reverse_samples(batch_size, intermediate_every = intermediate_every)
    intermediate_stats = compute_intermediate_stats(intermediates)
    intermediates = torch.cat(intermediates, dim = -1) # concat along width
    for idx, (image, intermediate_image) in enumerate(zip(
        torch.movedim(samples, 1, -1).cpu().numpy(),
        torch.movedim(intermediates, 1, -1).cpu().numpy()
    )):
        log_image = wandb.Image(image)
        log_intermediate = wandb.Image(intermediate_image)
        logs = {
            "reverse_examples": {
                "sample": log_image,
                "intermediates": log_intermediate,
            },
        }
        for stat_name, stat in intermediate_stats.items():
            logs["reverse_examples"][stat_name] = stat[idx]
        logger.add(logs, prefix = prefix)
    model.train()

@torch.no_grad()
def display_forward_samples(x_val, model, logger, intermediate_every = 200, prefix = "val"):
    model.eval()
    intermediates = model.forward_samples(x_val, intermediate_every = intermediate_every)
    intermediate_stats = compute_intermediate_stats(intermediates)
    intermediates = torch.cat(intermediates, dim = -1) # concat along width
    for idx, (image, intermediate_image) in enumerate(zip(
        torch.movedim(x_val, 1, -1).cpu().numpy(),
        torch.movedim(intermediates, 1, -1).cpu().numpy(),
    )):
        log_image = wandb.Image(image)
        log_intermediate = wandb.Image(intermediate_image)
        logs = {
            "forward_examples": {
                "image": log_image,
                "intermediates": log_intermediate,
            },
        }
        for stat_name, stat in intermediate_stats.items():
            logs["forward_examples"][stat_name] = stat[idx]
        logger.add(logs, prefix = prefix)
    model.train()

@torch.no_grad()
def display_graph_samples(batch_size, x_val, cond_val, model, logger, intermediate_every = 200, prefix = "val", eval_function = None, policy = "open_loop"):
    model.eval()
    # samples, intermediates = model.reverse_samples(batch_size, x_val, cond_val, intermediate_every = intermediate_every)
    masks = None
    info = {}
    if policy == "open_loop":
        samples, intermediates, _ = policies.open_loop(batch_size, model, x_val, cond_val, intermediate_every = intermediate_every)
    elif policy == "open_loop_clustered":
        samples, intermediates = policies.open_loop_clustered(batch_size, model, x_val, cond_val, intermediate_every = intermediate_every)
    elif policy == "open_loop_multi":
        samples, intermediates = policies.open_loop_multi(
            model, x_val, cond_val, num_attempts = 8, score_fn = lambda x: check_legality(x, x_val[0], cond_val.x, cond_val.is_ports, True)
            )
    elif policy == "iterative":
        samples, intermediates, masks, info = policies.iterative(
            model, x_val, cond_val, score_fn = lambda x, mask: check_legality(x, x_val[0], cond_val.x, mask, True) > 0.99    
        )
    else:
        raise NotImplementedError
    if masks is None:
        intermediate_images = [generate_batch_visualizations(inter, cond_val) for inter in intermediates]
    else:
        intermediate_images = [generate_batch_visualizations(inter, cond_val, mask) for inter, mask in zip(intermediates, masks)]
    intermediate_images = torch.cat(intermediate_images, dim = -1) # concat along width
    sample_images = generate_batch_visualizations(samples, cond_val)
    # should be a list of dicts, each dict corresponds to one sample
    eval_metrics = eval_function(samples, x_val, cond_val) if eval_function is not None else [{}] * batch_size
    
    for idx, (image, intermediate_image) in enumerate(zip(
        torch.movedim(sample_images, 1, -1).cpu().numpy(),
        torch.movedim(intermediate_images, 1, -1).cpu().numpy()
    )):
        log_image = wandb.Image(image)
        log_intermediate = wandb.Image(intermediate_image)
        logs = {
            "reverse_examples": {
                "sample": log_image,
                "intermediates": log_intermediate,
                **eval_metrics[idx],
                **info,
            },
        }
        logger.add(logs, prefix = prefix)
    model.train()
    return eval_metrics

@torch.no_grad()
def display_forward_graph_samples(x_val, cond_val, model, logger, intermediate_every = 200, prefix = "val"):
    model.eval()
    intermediates = model.forward_samples(x_val, cond_val, intermediate_every = intermediate_every)
    intermediate_stats = compute_intermediate_stats(intermediates)
    intermediate_images = [generate_batch_visualizations(inter, cond_val) for inter in intermediates]
    intermediate_images = torch.cat(intermediate_images, dim = -1) # concat along width
    x_images = generate_batch_visualizations(x_val, cond_val)
    for idx, (image, intermediate_image) in enumerate(zip(
        torch.movedim(x_images, 1, -1).cpu().numpy(),
        torch.movedim(intermediate_images, 1, -1).cpu().numpy(),
    )):
        log_image = wandb.Image(image)
        log_intermediate = wandb.Image(intermediate_image)
        logs = {
            "forward_examples": {
                "image": log_image,
                "intermediates": log_intermediate,
            },
        }
        for stat_name, stat in intermediate_stats.items():
            logs["forward_examples"][stat_name] = stat[idx]
        logger.add(logs, prefix = prefix)
    model.train()

@torch.no_grad()
def generate_report(num_samples, dataloader, model, logger, policy = "iterative", intermediate_every = 200):
    metrics = common.Metrics()
    for _ in range(num_samples):
        x_eval, cond_eval = dataloader.get_batch("val")
        x_eval = x_eval[:1]
        sample_metrics = display_graph_samples(1, x_eval, cond_eval, model, logger, prefix = "eval", eval_function = eval_samples, policy = policy, intermediate_every = intermediate_every)
        for sample_metric in sample_metrics:
            metrics.add(sample_metric)
        cond_eval.to(device = "cpu")
    # compile metrics and compute stats
    logger.add(metrics.result(), prefix = "eval")

@torch.no_grad()
def save_outputs(
    x_in, 
    cond, 
    model, 
    save_folder, 
    output_number_offset=0, 
    policy="open_loop", 
    policy_kwargs = {}, 
    preprocess_fn=None, 
    postprocess_fn=None, 
    legalization_fn=None,
    ): 
    """
    x_in and cond are both assumed to be on CPU
    x_in has shape (V, 2)
    preprocess_fn: x_in, cond -> x_in, cond
    postprocess_fn: sample, cond -> sample
    Returns:
    - metrics: Dict
    - sample: (V, 2) tensor 
    - cond: Data object
    All outputs are after preprocessing, and before postprocessing. tensors are on cpu
    """
    idx = cond.file_idx if "file_idx" in cond else output_number_offset
    x_in = torch.unsqueeze(x_in, dim=0).to(model.device)
    original_device = cond.x.device
    cond.to(model.device)
    metrics = {}
    metrics_special = {} # For things that should not be aggregated like plots, images, etc.

    # user-defined preprocess function
    t0 = time.time()
    x_preprocessed, cond_preprocessed = preprocess_fn(x_in, cond) if preprocess_fn is not None else (x_in, cond)

    t1 = time.time()
    if cond_preprocessed.num_nodes == 0:
        # handle edge case with 0 nodes after preprocessing
        sample = torch.zeros_like(x_preprocessed)
    else:
        if policy == "open_loop":
            sample, _, policy_metrics_special = policies.open_loop(1, model, x_preprocessed, cond_preprocessed, intermediate_every = 0, save_videos = policy_kwargs["save_videos"])
            metrics_special.update(policy_metrics_special)
        elif policy == "open_loop_clustered":
            sample, _ = policies.open_loop_clustered(1, model, x_preprocessed, cond_preprocessed, intermediate_every = 0)
        elif policy == "iterative_clustering":
            sample, policy_metrics, policy_metrics_special = policies.iterative_clustering(1, model, x_preprocessed, cond_preprocessed, **policy_kwargs)
            metrics.update(policy_metrics)
            metrics_special.update(policy_metrics_special)
        elif policy == "random":
            sample = policies.random(1, x_preprocessed, cond_preprocessed)
        else:
            raise NotImplementedError
    t2 = time.time()

    # save image too
    image = visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))

    # legalization
    if legalization_fn is not None:
        sample, legalization_metrics, legalization_metrics_special = legalization_fn(sample, cond_preprocessed)
        metrics.update(legalization_metrics)
        metrics_special.update(legalization_metrics_special)
        image_legalized = visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
    else:
        image_legalized = image
    debug_plot_img(image_legalized, os.path.join(save_folder, f"placed{idx}"))

    # user-defined postprocess function
    sample_unprocessed = sample.detach().clone()
    sample, cond_postprocessed = postprocess_fn(sample, cond_preprocessed)

    sample = sample.squeeze(dim=0).detach().to(device = cond.x.device)
    sample = postprocess_placement(sample, cond_postprocessed).cpu().numpy() # mandatory post-processing
    save_file = os.path.join(save_folder, f"sample{idx}.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(sample, f)
    t3 = time.time()
    
    # evaluate sample and generate sampling metrics
    hpwl_normalized, hpwl_rescaled = hpwl_fast(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    macro_hpwl_normalized, macro_hpwl_rescaled = macro_hpwl(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    legality = check_legality_new(sample_unprocessed[0], x_in[0], cond_preprocessed, cond_preprocessed.is_ports, score=True)
    if "is_macros" in cond:
        macro_legality = check_legality_new(sample_unprocessed[0], x_in[0], cond_preprocessed, (~cond_preprocessed.is_macros) | cond_preprocessed.is_ports, score=True)
    else:
        macro_legality = 0.0
    original_hpwl_normalized = hpwl_fast(x_preprocessed, cond_preprocessed, normalized_hpwl=True)
    t4 = time.time()

    cond.to(original_device)

    metrics.update({
        "idx": idx,
        "hpwl_normalized": hpwl_normalized,
        "hpwl_rescaled": hpwl_rescaled,
        "macro_hpwl_normalized": macro_hpwl_normalized,
        "macro_hpwl_rescaled": macro_hpwl_rescaled,
        "legality_2": legality,
        "macro_legality": macro_legality,
        "original_hpwl_normalized": original_hpwl_normalized,
        "hpwl_ratio": hpwl_normalized/max(1e-12, original_hpwl_normalized),
        "model_time": t2-t1,
        "generation_time": t3-t0,
        "eval_time": t4-t3,
        "model_vertices": cond_preprocessed.num_nodes, # number of vertices that model input has
        "model_edges": cond_preprocessed.num_edges, # number of edges that model input has
    })
    return metrics, metrics_special, image, image_legalized

def compute_intermediate_stats(intermediates):
    # input: intermediates is a list, each is (B, C, H, W)
    # outputs: dict of stats, each value is torch tensor with shape (B, T)
    stats_to_compute = {"mean": torch.mean, "std": torch.std}
    stats = {}
    for stat_name, stat_fn in stats_to_compute.items():
        stat_list = [stat_fn(image.view(image.shape[0], -1), dim=1) for image in intermediates]
        stat = torch.cat(stat_list, -1)
        stats[stat_name] = stat
    return stats

def eval_samples(samples, x_val, cond_val, use_new_legality_fn = True):
    # evaluates generated samples
    # returns a list (length B) of dicts, each dict corresponds to one sample
    # samples and x_val are (B, V, F)
    eval_metrics = []
    cond_ports = cond_val.is_ports
    for idx, (sample, x) in enumerate(zip(samples, x_val)):
        V, F = sample.shape
        sample_hpwl = hpwl(sample, cond_val)
        original_hpwl = hpwl(x, cond_val)
        current_metrics = {
            "num_vertices": V,
            "num_edges": cond_val.edge_index.shape[1],
            # "legality_score": check_legality(sample, x, cond_val, cond_ports, score=True), # Deprecated! don't use this
            # "is_legal": check_legality(sample, x, cond_val, cond_ports, score=False),
            "gen_hpwl": sample_hpwl,
            "original_hpwl": original_hpwl,
            "hpwl_ratio": sample_hpwl/original_hpwl if original_hpwl!=0 else 0,
        }
        if use_new_legality_fn:
            current_metrics["legality_score_2"] = check_legality_new(sample, x, cond_val, cond_ports, score=True)
        eval_metrics.append(current_metrics)
    return eval_metrics

def load_graph_data(dataset_name, augment = False, train_data_limit = None, val_data_limit = None):
    dataset_sizes = { # name: (train size, val size, chip width, chip height, scale)
    }
    dataset_path = os.path.join(os.path.dirname(__file__), f'../datasets/graph/{dataset_name}')
    if os.path.exists(dataset_path):
        if dataset_name in dataset_sizes:
            TRAIN_SIZE, VAL_SIZE, chip_width, chip_height, scale = dataset_sizes[dataset_name]
        else:
            return load_graph_data_with_config(dataset_name, train_data_limit=train_data_limit, val_data_limit=val_data_limit)
        if train_data_limit is None or train_data_limit == "none":
            train_data_limit = TRAIN_SIZE
        if val_data_limit is None or val_data_limit == "none":
            val_data_limit = VAL_SIZE
        assert train_data_limit <= TRAIN_SIZE and val_data_limit <= VAL_SIZE, "data limits invalid"
        train_set = []
        val_set = []
        missing_data = 0
        for i in range(TRAIN_SIZE + VAL_SIZE):
            if not (i<train_data_limit or (i>=TRAIN_SIZE and i-TRAIN_SIZE<val_data_limit)):
                continue
            cond_path = os.path.join(dataset_path, f"graph{i}.pickle")
            x_path = os.path.join(dataset_path, f"output{i}.pickle")
            if not (os.path.exists(cond_path) and os.path.exists(dataset_path)):
                missing_data += 1
                if missing_data <= 5:
                    print(f"WARNING: {i} of dataset not found in {dataset_path}")
                if missing_data == 5:
                    print(f"Suppressing missing data warnings...")
                continue
            cond = load_and_parse_graph(cond_path)
            x = open_pickle(x_path)
            x, cond = preprocess_graph(x, cond, (chip_width, chip_height), scale)
            if i<TRAIN_SIZE:
                train_set.append((x, cond))
            else:
                val_set.append((x, cond))
        if missing_data > 0:
            print(f"WARNING: total of {missing_data} samples not found. Continuing...")
    else:
        try:
            return load_synthetic_graph_data(dataset_name, train_data_limit, val_data_limit)
        except NotImplementedError:
            raise
    return train_set, val_set

def load_graph_data_with_config(dataset_name, train_data_limit = None, val_data_limit = None, override_placement_path = None, placement_format = "output*.pickle"):
    # loads data in a way that maintains link with original files
    # Algorithm:
    # load and parse config
    # sort all files (graphXX and outputXX) in increasing order of number XX
    # the first TRAIN_SIZE examples are in the training set, next VAL_SIZE are in test set
    # generate a list of filenames along with placement and netlist

    dataset_path = os.path.join(os.path.dirname(__file__), f'../datasets/graph/{dataset_name}')
    placement_path = dataset_path if override_placement_path is None else override_placement_path
    if os.path.exists(dataset_path):
        config = get_dataset_config(dataset_name)
        TRAIN_SIZE = config.train_samples
        VAL_SIZE = config.val_samples
        scale = config.scale

        if train_data_limit is None or train_data_limit == "none":
            train_data_limit = TRAIN_SIZE
        if val_data_limit is None or val_data_limit == "none":
            val_data_limit = VAL_SIZE
        assert train_data_limit <= TRAIN_SIZE and val_data_limit <= VAL_SIZE, "data limits invalid"
        
        graph_files = {int(re.search('\d+', p.name).group()):str(p) for p in Path(dataset_path).rglob("graph*.pickle")}
        placement_files = {int(re.search('\d+', p.name).group()):str(p) for p in Path(placement_path).rglob(placement_format)}
        idx_list = list(graph_files.keys())
        idx_list.sort()
        intersect_list = list(graph_files.keys() & placement_files.keys())
        if len(intersect_list) != len(idx_list):
            print(f"WARNING: some graph files have no corresponding placements. {len(intersect_list)} of {len(idx_list)} placements found.")

        assert TRAIN_SIZE + VAL_SIZE <= len(idx_list), "not enough valid data files found for TRAIN_SIZE and VAL_SIZE specified"
        train_idx = idx_list[:train_data_limit]
        val_idx = idx_list[TRAIN_SIZE:TRAIN_SIZE + val_data_limit]

        train_set = []
        val_set = []
        for train_i in train_idx:
            cond_path = graph_files[train_i]
            x_path = placement_files.get(train_i, None)
            cond = load_and_parse_graph(cond_path)
            if x_path is not None:
                x = open_pickle(x_path)
            else:
                x = torch.zeros_like(cond.x)
            if "chip_size" in cond:
                if len(cond.chip_size) == 4: # chip_size is [x_start, y_start, x_end, y_end]
                    chip_size = (cond.chip_size[2] - cond.chip_size[0], cond.chip_size[3] - cond.chip_size[1])
                    chip_offset = (cond.chip_size[0], cond.chip_size[1])
                else:
                    chip_size = (cond.chip_size[0], cond.chip_size[1])
                    chip_offset = (0, 0)
            else:
                chip_size = (config.chip_width, config.chip_height)
                chip_offset = (0, 0)
            x, cond = preprocess_graph(x, cond, chip_size, scale, chip_offset=chip_offset)
            cond.file_idx = train_i
            train_set.append((x, cond))
        for val_i in val_idx:
            cond_path = graph_files[val_i]
            x_path = placement_files.get(val_i, None)
            cond = load_and_parse_graph(cond_path)
            if x_path is not None:
                x = open_pickle(x_path)
            else:
                x = torch.zeros_like(cond.x)
            if "chip_size" in cond: 
                if len(cond.chip_size) == 4: # chip_size is [x_start, y_start, x_end, y_end]
                    chip_size = (cond.chip_size[2] - cond.chip_size[0], cond.chip_size[3] - cond.chip_size[1])
                    chip_offset = (cond.chip_size[0], cond.chip_size[1])
                else:
                    chip_size = (cond.chip_size[0], cond.chip_size[1])
                    chip_offset = (0, 0)
            else:
                chip_size = (config.chip_width, config.chip_height)
                chip_offset = (0, 0)
            x, cond = preprocess_graph(x, cond, chip_size, scale, chip_offset=chip_offset)
            cond.file_idx = val_i
            if dataset_name == "ispd2005": # TODO fix special case
                cond.is_ports = torch.zeros_like(cond.is_ports)
            val_set.append((x, cond))
    else:
        try:
            return load_synthetic_graph_data(dataset_name, train_data_limit, val_data_limit)
        except NotImplementedError:
            raise
    return train_set, val_set

def get_dataset_config(dataset_name):
    # raises error if file is not found
    dataset_path = os.path.join(os.path.dirname(__file__), f'../datasets/graph/{dataset_name}')
    if os.path.exists(dataset_path):
        config_path = os.path.join(dataset_path, "config.yaml")
        config = OmegaConf.load(config_path)
        return config
    else:
        raise FileNotFoundError

def load_synthetic_graph_data(dataset_name, train_data_limit = None, val_data_limit = None):
    dataset_path = os.path.join(os.path.dirname(__file__), '../data-gen/outputs', dataset_name)
    NEEDS_CENTERING = False
    # load dataset config
    config_path = list(Path(dataset_path).glob("config.yaml"))
    assert len(config_path) > 0, f"config path not found in {dataset_path}"

    dataset_config = OmegaConf.load(config_path[0])
    if ("num_train_samples" in dataset_config and "num_val_samples" in dataset_config):
        TRAIN_SIZE = dataset_config.num_train_samples
        VAL_SIZE = dataset_config.num_val_samples
        data_files = {int(re.search('\d+', p.name).group()):str(p) for p in Path(dataset_path).rglob("*.pickle")}
        data_files = OrderedDict(sorted(data_files.items()))
        if train_data_limit is None or train_data_limit == "none":
            train_data_limit = TRAIN_SIZE
        if val_data_limit is None or val_data_limit == "none":
            val_data_limit = VAL_SIZE
        assert train_data_limit <= TRAIN_SIZE and val_data_limit <= VAL_SIZE, "data limits invalid"
        train_set = []
        val_set = []
        for _, data_file in data_files.items():
            x = open_pickle(data_file)
            
            new_len = len(x)
            train_len = min(train_data_limit-len(train_set), new_len)
            val_len = min(val_data_limit-len(val_set), new_len-train_len)
            
            train_samples = x[:train_len]
            val_samples = x[train_len:train_len+val_len]
            train_samples = [preprocess_synthetic_graph(x, cond, NEEDS_CENTERING) for x, cond in train_samples] 
            val_samples = [preprocess_synthetic_graph(x, cond, NEEDS_CENTERING) for x, cond in val_samples]
            train_set.extend(train_samples)
            val_set.extend(val_samples)
            
            if len(train_set) == train_data_limit and len(val_set) == val_data_limit:
                break
    else:
        # dataset is a mixture of other datasets
        data_mix = dataset_config.mixture
        train_set = []
        val_set = []
        if train_data_limit is None or train_data_limit == "none":
            train_data_limit = int(1e16)
        else:
            print("WARNING: mixture train set will truncate without shuffling, because train_data_limit is defined")
        if val_data_limit is None or val_data_limit == "none":
            val_data_limit = int(1e16)
        else:
            print("WARNING: mixture val set will truncate without shuffling, because val_data_limit is defined")
        for k, v in data_mix.items(): # be careful of recursion
            train_samples, val_samples = load_synthetic_graph_data(
                k, 
                train_data_limit = v.num_train_samples, 
                val_data_limit = v.num_val_samples,
                )
            train_set.extend(train_samples)
            val_set.extend(val_samples)
            if len(train_set) >= train_data_limit and len(val_set) >= val_data_limit:
                break
        train_set = train_set[:train_data_limit]
        val_set = val_set[:val_data_limit]
    return train_set, val_set

def load_samples(samples_dir):
    """
    Loads placements for generated samples, as well as the graphs
    Then preprocesses them, and outputs the preprocessed placements and graphs
    """
    # load graphs used to generate samples
    samples_config_path = str(list(pathlib.Path(samples_dir).rglob("config.yaml"))[0])
    cfg = OmegaConf.load(samples_config_path)
    _, val_set = load_graph_data_with_config(
        cfg.task, 
        train_data_limit = cfg.train_data_limit, 
        val_data_limit = cfg.val_data_limit,
        override_placement_path = samples_dir,
        placement_format = "sample*.pkl",
        )
    return val_set

def preprocess_synthetic_graph(x, cond, needs_centering=False):
    """
    Preprocesses synthetic data.
    Terminal positions are measured relative to center of instance, and normalized to canvas size
    """
    if needs_centering:
        x = x + cond.x/2
        u_shape = cond.x[cond.edge_index[0,:]]
        v_shape = cond.x[cond.edge_index[1,:]]
        cond.edge_attr[:,:2] = cond.edge_attr[:,:2] - u_shape/2
        cond.edge_attr[:,2:4] = cond.edge_attr[:,2:4] - v_shape/2
    assert (cond.edge_index.shape[1] % 2 == 0) and (cond.edge_attr.shape[0] % 2 == 0), "graph edges must be undirected"
    return x, cond

def preprocess_graph(x, cond, chip_size, scale = 1, chip_offset = (0, 0)):
    # x: numpy float64 array with shape (V, 2) describing 2D position on canvas
    # cond.x: torch float32 tensor (V, 2) describing instance sizes
    # cond.edge_attr: torch float64 tensor (E, 4)
    # chip_size: tuple of length 2; size of canvas in um
    # chip_size: tuple of length 2; bottom left corner coordinates in um

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

def edge_dropout(x, cond, dropout_probability):
    _, E = cond.edge_index.shape
    E = E//2 # forward and reverse edges are included in edge_index. we only want forward edges
    num_remaining_edges = round((1-dropout_probability) * E)
    if num_remaining_edges > 0:
        new_edge_ids = torch.multinomial(torch.ones([E], device=cond.edge_index.device), num_remaining_edges) # sample without replacement
    else:
        new_edge_ids = torch.tensor([0], dtype=torch.int64) # need at least one edge
    new_edge_ids = torch.cat([new_edge_ids, new_edge_ids + E])
    
    output_cond = Data(
        x = cond.x,
        edge_index = cond.edge_index[:, new_edge_ids],
        edge_attr = cond.edge_attr[new_edge_ids, :],
    )
    if "edge_pin_id" in cond:
        output_cond.edge_pin_id = cond.edge_pin_id[new_edge_ids, :]
    # (shallow) copy over other attributes in cond
    for k in cond.keys():
        if not k in output_cond:
            output_cond[k] = cond[k]
    return x, output_cond

def generate_batch_visualizations(x, cond, mask = None):
    # x has shape (B, V, 2)
    # cond is data object, cond.x contains width and heights of nodes
    # mask is shared across batch dimension, has shape (V)
    B, V, F = x.shape
    image_list = []
    for i in range(B):
        img = torch.tensor(visualize_placement(x[i], cond, mask = cond.is_ports if mask is None else mask)).movedim(-1, -3) # images should be C, H, W
        C, H, W = img.shape
        img_padded = torch.zeros((C, H+2, W+2), dtype=img.dtype, device=img.device)
        img_padded[:, 1:-1, 1:-1] = img
        image_list.append(img_padded)
    return torch.stack(image_list, dim=0)

class DataLoader:
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            train_batch_size,
            val_batch_size,
            train_device = "cuda",
            num_workers = 8,
            pin_memory = False,
        ):
        self.device = train_device
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_set = train_dataset
        self.val_set = val_dataset
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, pin_memory_device=train_device if pin_memory else '')
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, pin_memory_device=train_device if pin_memory else '')
        self._get_train_batch = self._train_gen()
        self._get_val_batch = self._val_gen()
        self._display_x = None
        self._display_y = None

    def get_batch(self, split):
        assert split in ("train", "val"), "split argument has to be one of 'train' or 'val'"
        x, y = next(self._get_train_batch) if split == "train" else next(self._get_val_batch)
        return x.to(self.device), y.to(self.device)
    
    def get_display_batch(self, num_images):
        assert num_images <= self.val_batch_size, "num images must be smaller than batch size"
        if (self._display_x is None) or (self._display_y is None):
            x, y = self.get_batch("val")
            self._display_x = x[:num_images]
            self._display_y = y[:num_images]
        return self._display_x, self._display_y

    def _train_gen(self):
        while True:
            for data in self.train_loader:
                yield data
    
    def _val_gen(self):
        while True:
            for data in self.val_loader:
                yield data
    
    def get_train_size(self):
        return len(self.train_set)

    def get_val_size(self):
        return len(self.val_set)
    
class GraphDataLoader:
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            train_batch_size,
            val_batch_size,
            train_device = "cuda",
            preprocess_fn = None,
            train_shuffle = True,
            val_shuffle = True,
            num_workers = 8,
            pin_memory = False,
        ):
        self.device = train_device
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_set = train_dataset
        self.val_set = val_dataset
        self._display_x = {}
        self._display_y = {}
        self.preprocess_fn = preprocess_fn

        self.is_shuffle = {"train": train_shuffle, "val": val_shuffle}
        self.current_idx = {"train": 0, "val": 0} # For non-shuffle
    
    def get_batch(self, split):
        assert split in ("train", "val"), "split argument has to be one of 'train' or 'val'"
        dataset = self.train_set if split=="train" else self.val_set
        batch_size = self.train_batch_size if split=="train" else self.val_batch_size
        
        if self.is_shuffle[split]:
            idx = torch.randint(0, len(dataset), [1]) # TODO support larger batch sizes
        else:
            idx = self.current_idx[split]
            self.current_idx[split] = (self.current_idx[split] + 1) % len(dataset)
        
        x, y = dataset[idx]
        output = self.prepare_output(x.to(self.device).view(1, *x.shape).expand(batch_size, *x.shape), y.to(self.device))
        return output

    def get_display_batch(self, display_batch_size, split="val"):
        batch_size = self.val_batch_size if split == "val" else self.train_batch_size
        assert display_batch_size <= batch_size, "num images must be smaller than batch size"
        if (self._display_x.get(split, None) is None) or (self._display_y.get(split, None) is None):
            x, y = self.get_batch(split)
            # self._display_x[split] = x[:display_batch_size]
            # self._display_y[split] = y
        # return self._display_x[split], self._display_y[split]
        output = self.prepare_output(x[:display_batch_size], y)
        return output # TODO return deterministically

    def reset_idx(self, split):
        assert split in ("train", "val"), "split argument has to be one of 'train' or 'val'"
        self.current_idx[split] = 0

    def get_train_size(self):
        return len(self.train_set)

    def get_val_size(self):
        return len(self.val_set)

    def prepare_output(self, x, y):
        if self.preprocess_fn is None:
            return x, y
        else:
            return self.preprocess_fn(x, y)

def open_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_and_parse_graph(path):
    """Loads Data object from pickle file"""
    graph = open_pickle(path) # networkx graph or pytorch geometric
    assert isinstance(graph, Data), "Pickle file must contain pytorch-geometric Data object; networkx is deprecated"
    assert (graph.edge_index.shape[1] % 2 == 0) and (len(graph.edge_attr) % 2 == 0), "graph edges must be undirected"
    # enforce 1D tensor for some keys
    for key in ["is_macros", "is_ports"]:
        if key in graph and len(graph[key].shape) > 1:
            graph[key] = torch.flatten(graph[key]).bool() # TODO remove autoconversions to avoid bugs
    return graph

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

def visualize(x, cond, mask = None):
    """ 
    Visualizes the X with node attributes, returning an numpy image (H, W, C)
    x,y are floats normalized to canvas size (from -1 to 1)
    attr are also normalized to canvas size
    Inputs:
    x: can be (V, 2) for 2D coordinate placement or (V, 2+3) floats for 2D plus orientation placement
    cond: pyG data object with instance sizes in 'x' field (V, 2)
    """
    width, height = 128, 128
    background_color = "white"
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    assert len(x.shape) == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    if x.shape[1] == 5:
        # 2D + orientation placement
        cond = orientations.to_fixed(x[:,2:], cond)
    attr = cond.x.cpu()
    x = x[:,:2].cpu()

    h_step = 1 / len(x)
    for i, (pos, shape) in enumerate(zip(x, attr)):
        # NOTE assumes coordinates are center of instance
        left = pos[0] - shape[0]/2
        top = pos[1] + shape[1]/2
        right = pos[0] + shape[0]/2
        bottom = pos[1] - shape[1]/2
        inbounds = (left>=-1) and (top<=1) and (right<=1) and (bottom>=-1)

        left = (0.5 + left/2) * width
        right = (0.5 + right/2) * width
        top = (0.5 - top/2) * height
        bottom = (0.5 - bottom/2) * height

        color = hsv_to_rgb(i * h_step, 1 if (mask is None or not mask[i]) else 0.2, 0.9 if inbounds else 0.5)
        draw.rectangle([left, top, right, bottom], fill=color, width=0)

    return np.array(image)

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
    if x.shape[1] == 5:
        # 2D + orientation placement
        cond = orientations.to_fixed(x[:,2:], cond)
    x = x[:,:2]

    def canvas_to_pixel_coord(x):
        # x is (B, 2) tensor representing normalized 2D coordinates in canvas space
        output = torch.zeros_like(x)
        output[:,0] = (0.5 + x[:,0]/2) * width
        output[:,1] = (0.5 - x[:,1]/2) * height
        return output

    V, _ = x.shape
    mask = cond.is_ports if "is_ports" in cond and mask is None else mask
    h_step = 0.2 / max(V, 1) if "is_macros" in cond else 1.0 / max(V, 1)
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

def save_cfg(cfg, path):
    with open(path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

def load_cfg(path):
    with open(path, "r") as f:
        return OmegaConf.load(f)

def visualize_ignore_ports(x, cond, mask):
    """ 
    Visualizes the X with node attributes, returning an numpy image
    x,y are floats normalized to canvas size (from -1 to 1)
    attr are also normalized to canvas size
    """
    width, height = 1024, 1024
    background_color = "black"
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    assert len(x.shape) == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    if x.shape[1] == 5:
        # 2D + orientation placement
        cond = orientations.to_fixed(x[:,2:], cond)
    attr = cond.x.cpu()
    x = x[:,:2].cpu()

    # h_step = 1 / len(x)
    for i, (pos, shape) in enumerate(zip(x, attr)):
        if not mask[i]:
            # NOTE assumes coordinates are center of instance
            left = pos[0] - shape[0]/2
            top = pos[1] + shape[1]/2
            right = pos[0] + shape[0]/2
            bottom = pos[1] - shape[1]/2
            inbounds = (left>=-1) and (top<=1) and (right<=1) and (bottom>=-1)

            left = (0.5 + left/2) * width
            right = (0.5 + right/2) * width
            top = (0.5 - top/2) * height
            bottom = (0.5 - bottom/2) * height
            color = (255, 255, 255) if inbounds else (0, 0, 0)
            draw.rectangle([left, top, right, bottom], fill=color, width=0)

    # image.save("debug.png")
    # import pdb; pdb.set_trace()

    return np.array(image)

def hpwl(samples, cond_val):
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
    if samples.shape[1] == 5:
        # 2D + orientation placement
        cond_val = orientations.to_fixed(samples[:,2:], cond_val)
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
    # TODO scale by x and y if necessary
    hpwl = sum([(max(n[::2]) - min(n[::2])) + (max(n[1::2]) - min(n[1::2])) for n in nets.values()])
    return hpwl

def hpwl_fast(x, cond, normalized_hpwl = True):
    """
    Returns hpwl computed using custom GNN trick
    If not normalized_hpwl, will return both normalized HPWL, as well as rescaled HPWL (using original units)
    """
    hpwl_net = guidance.HPWL()
    pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond)
    hpwl_net = hpwl_net(x, pin_map, pin_offsets, pin_edge_index, net_aggr="sum", raw_output = (not normalized_hpwl))
    if normalized_hpwl:
        return hpwl_net.item() # output is hpwl, no additional processing needed
    else:
        if "chip_size" in cond:
            x_scale = (cond.chip_size[2] - cond.chip_size[0])/2 # because chip is from [-1, 1] when normed
            y_scale = (cond.chip_size[3] - cond.chip_size[1])/2
        else:
            x_scale = 1
            y_scale = 1
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
    if not ("is_macros" in cond):
        return 0 if normalized_hpwl else (0, 0)

    hpwl_net = guidance.MacroHPWL()
    pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond)
    hpwl_net = hpwl_net(x, pin_map, pin_offsets, pin_edge_index, cond.is_macros, net_aggr="sum", raw_output = (not normalized_hpwl))
    if normalized_hpwl:
        return hpwl_net.item() # output is hpwl, no additional processing needed
    else:
        if "chip_size" in cond:
            x_scale = (cond.chip_size[2] - cond.chip_size[0])/2 # because chip is from [-1, 1] when normed
            y_scale = (cond.chip_size[3] - cond.chip_size[1])/2
        else:
            x_scale = 1
            y_scale = 1
        scale_factor = torch.tensor([[x_scale, y_scale, x_scale, y_scale]]).to(device = hpwl_net.device)
        rescaled_hpwl = (scale_factor * hpwl_net).sum(dim=-1).sum(dim=-1) 
        norm_hpwl = hpwl_net.sum(dim=-1).sum(dim=-1)
        return norm_hpwl.item(), rescaled_hpwl.item()

def edge_length(x, cond):
    """
    Returns total L2 norm of edges in graph 
    """
    unique_edges = cond.edge_attr.shape[0]//2
    u_pos = cond.edge_attr[:unique_edges,:2] + x[cond.edge_index[0,:unique_edges]]
    v_pos = cond.edge_attr[:unique_edges,2:4] + x[cond.edge_index[1,:unique_edges]]
    edge_length = torch.linalg.vector_norm(u_pos - v_pos, dim=1).sum()
    return edge_length

def check_legality(x, y, cond, mask, score=True):
    # x is predicted placements (V, 2) or (V, 2+3)
    # y is ground truth placements (V, 2) TODO make it so that we don't actually need these
    # attr is width height (V, 2)
    # returns float with legality of placement (0 = bad, 1 = legal)
    # assert len(x.shape) == 2 and x.shape[1] == 2 and x.shape == y.shape
    
    assert len(x.shape) == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    cond_original = cond
    if x.shape[1] == 5:
        # 2D + orientation placement
        cond = orientations.to_fixed(x[:,2:], cond)
    attr = cond.x.cpu()
    x = x[:,:2].cpu()

    if not score:
        legal = 1
        width, height = 256, 256
        for i, (pos1, shape1) in enumerate(zip(x, attr)):
            for j, (pos2, shape2) in enumerate(zip(x, attr)):
                if (i != j):
                    # NOTE assumes coordinates are center of instance
                    # import pdb; pdb.set_trace()
                    left1 = round(float(pos1[0] - shape1[0]), 3)
                    top1 = round(float((pos1[1] + shape1[1])), 3)
                    right1 = round(float((pos1[0] + shape1[0])), 3)
                    bottom1 = round(float(pos1[1] - shape1[1]), 3)

                    left1 = (0.5 + left1/2) * width
                    right1 = (0.5 + right1/2) * width
                    top1 = (0.5 - top1/2) * height
                    bottom1 = (0.5 - bottom1/2) * height

                    left2 = round(float(pos2[0] - shape2[0]), 3)
                    top2 = round(float((pos2[1] + shape2[1])), 3)
                    right2 = round(float((pos2[0] + shape2[0])), 3)
                    bottom2 = round(float(pos2[1] - shape2[1]), 3)

                    left2 = (0.5 + left2/2) * width
                    right2 = (0.5 + right2/2) * width
                    top2 = (0.5 - top2/2) * height
                    bottom2 = (0.5 - bottom2/2) * height
                    try:
                        rectangle1 = Polygon([(left1, top1), (left1, bottom1), (right1, bottom1), (right1, top1)])
                        rectangle2 = Polygon([(left2, top2), (left2, bottom2), (right2, bottom2), (right2, top2)])
                    except Exception as e:
                        print(e)
                        return 0

                    if rectangle1.intersects(rectangle2) and not rectangle1.touches(rectangle2) and not mask[i] and not mask[j]:
                        legal = 0
        return legal
    else:
        placement = visualize_ignore_ports(x, cond, mask)
        reference = visualize_ignore_ports(y, cond_original, mask)
        if np.count_nonzero(reference[:,:,0]) == 0: # divide by 0 iminent
            import ipdb; ipdb.set_trace()
            return 0
        return np.count_nonzero(placement[:,:,0]) / np.count_nonzero(reference[:,:,0])

def check_legality_new(x, y, cond, mask, score=True):
    # x is predicted placements (V, 2)
    # y is ground truth placements (V, 2) -- not used
    # attr is width height (V, 2)
    # returns float with legality of placement (1 = legal, 0 = bad)
    if mask.sum() == cond.num_nodes: # everything is masked out
        return 1.0
    assert len(x.shape) == 2, "x has to have 2 axes with shape (V, 2) or (V, 2+3)"
    if x.shape[1] == 5:
        # 2D + orientation placement
        cond = orientations.to_fixed(x[:,2:], cond)
    attr = cond.x.cpu().numpy()
    x = x[:,:2].cpu().numpy()
    mask = mask.cpu().numpy()

    # if nans are found, just give up
    if np.isnan(np.sum(x)):
        return 0 if score else False

    insts = [shapely.box(loc[0] - size[0]/2, loc[1] - size[1]/2, loc[0] + size[0]/2, loc[1] + size[1]/2) 
                for size, loc, is_ports in zip(attr, x, mask)
                if not is_ports]
    chip = shapely.box(-1, -1, 1, 1)
    insts_area = round(sum([i.area for i in insts]), 6)
    insts_union = round(shapely.intersection(shapely.unary_union(insts), chip).area, 6)
    if score:
        return insts_union/insts_area
    else:
        return insts_union >= insts_area

def get_hyperedges(cond_val):
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

def cluster(input_cond, num_clusters, temp_dir = "logs/temp", verbose = False, placements = None, shmetis_dir = "."):
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
    
    salt = np.random.randint(1e12) # so we can run operations in parallel
    hmetis_input_file = os.path.join(temp_dir, f'edges-{salt}.txt')
    
    # input file
    # first line is number of hyperedges and number of vertices
    # i th line (excluding comment lines) contains the vertices that are included in the (i−1)th hyperedge
    with open(hmetis_input_file, 'w') as fp:
        fp.write(f'{len(hyp_e)} {max([n for k in hyp_e for n in hyp_e[k]])}\n')
        for k in hyp_e:
            fp.write(' '.join(map(str, hyp_e[k])) + '\n')

    # call hmetis and parse output
    # shmetis HGraphFile Nparts UBfactor
    # TODO set UBfactor as setting
    shmetis_path = str(os.path.join(shmetis_dir, "shmetis"))
    subprocess.run([shmetis_path, hmetis_input_file, str(num_clusters), str(5)], capture_output = not verbose)

    hmetis_output_file = f'{hmetis_input_file}.part.{num_clusters}'
    with open(hmetis_output_file, 'r') as fp:
        assigned_parts = list(map(int, fp.readlines()))
    assert len(assigned_parts) == input_cond.x.shape[0], f"error parsing shmetis output. expected lines {len(assigned_parts)} but got {input_cond.x.shape[0]}"

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
    output_cond["original_cond"] = input_cond

    # return to original device
    placements.to(device = placements_device)
    input_cond.to(device = cond_device)

    # clean up hmetis i/o files
    os.remove(hmetis_input_file)
    os.remove(hmetis_output_file)
    
    return output_cond.to(device = cond_device), output_placements.to(device = placements_device)

def uncluster(clustered_cond, clustered_x, return_cond = False):
    """
    generate original placements based on clustered cond and placements
    clustered_cond contains cluster_map, which is a (V_unclustered) tensor
    map: unclustered_id -> clustered_id
    clustered_x: (B, V, 2)
    returns x (B, V_unclustered, 2)
    """
    unclustered_x = clustered_x[:, clustered_cond.cluster_map, :]
    if return_cond:
        if "original_cond" in clustered_cond:
            output_cond = clustered_cond.original_cond
        else:
            output_cond = None
        return unclustered_x, output_cond
    else:
        return unclustered_x

def remove_non_macros(x, cond, remove_mask = None):
    """
    Given placement and graph (cond),
    Removes non-macros (ports and SC clusters) so that only macros remain
    If a net has SC as source, will assign new source (randomly | largest area)
    Inputs:
    - remove_mask: (V,) bool tensor. True if keep objects, False if remove
    Returns new placement and cond, while storing old data in the cond
    """
    device = cond.x.device
    remove_mask = cond.is_macros if remove_mask is None else remove_mask # True if keep entries, False if remove
    E = cond.num_edges
    edge_remove_mask = remove_mask[cond.edge_index][:, :E//2] # (2, E//2)

    pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond) # (P_u,),(P_u,2),(2,P)
    
    # remove invalid destinations
    pin_edge_index = pin_edge_index[:, edge_remove_mask[1, :]] # (2, P)
    edge_remove_mask = remove_mask[pin_map[pin_edge_index]] # (2, P)
    
    # remove invalid sources
    nets = {}
    for i in range(pin_edge_index.shape[-1]):
        if edge_remove_mask[0, i]: # keep entry, no-op
            continue
        src_id = pin_edge_index[0, i].cpu().item()
        dest_id = pin_edge_index[1, i].cpu().item()
        nets[src_id] = [dest_id] if src_id not in nets else nets[src_id] + [dest_id]
    
    # assign new sources
    max_pins = pin_edge_index.max()+1 if pin_edge_index.numel()>0 else 1
    src_pin_map = torch.arange(0, max_pins, device=device)
    for src_id, dests in nets.items():
        # TODO implement other source-assignment methods like largest area
        new_src = dests[torch.randint(len(dests),(1,)).item()]
        src_pin_map[src_id] = new_src
    pin_edge_index[0, :] = src_pin_map[pin_edge_index[0, :]] # (2, P)

    # remove self-edges
    pin_edge_index = pin_edge_index[:, pin_edge_index[0]!=pin_edge_index[1]]

    # reconstruct edge_index and edge_attr
    pin_edge_index = torch.concat((pin_edge_index, torch.flip(pin_edge_index,dims=(0,))), dim=-1)
    output_edge_index_oldlabels = pin_map[pin_edge_index]

    output_edge_attr = pin_offsets[pin_edge_index.T, :] # (E, src/dest, x/y)
    output_edge_attr = output_edge_attr.reshape(pin_edge_index.shape[1], 4)

    # relabel the nodes
    vertex_map = torch.arange(0, cond.num_nodes, device=device)[remove_mask]
    new_indices = torch.cumsum(remove_mask, dim=0)-1
    output_edge_index = new_indices[output_edge_index_oldlabels]
    assert output_edge_index.numel()==0 or output_edge_index.min()>=0
    output_cond = Data(
        x = cond.x[remove_mask, :],
        is_ports = cond.is_ports[remove_mask],
        is_macros = cond.is_macros[remove_mask],
        edge_index = output_edge_index,
        edge_attr = output_edge_attr,
        vertex_map = vertex_map, # original_node_idx = vertex_map[new_node_idx]
    )
    if "edge_pin_id" in cond:
        output_cond.edge_pin_id = pin_edge_index.T
    for k in cond.keys():
        if not k in output_cond:
            output_cond[k] = cond[k]
    output_cond.remove_tensor("cluster_map") # cluster map is meaningless
    output_cond["original_cond"] = cond
    return x[..., remove_mask, :], output_cond

def add_non_macros(x, cond):
    """
    Inverse of remove_non_macros
    Returns:
    x, cond
    """
    device = cond.x.device
    output_cond = cond.original_cond
    V = output_cond.num_nodes
    output_x = torch.zeros((*x.shape[:-2], V, x.shape[-1]), dtype = x.dtype, device = device)
    output_x[..., cond.vertex_map, :] = x
    return output_x, output_cond

def save_video(frames, out_path, fps=15):
    """
    Saves video to disk
    Frames is list(array(H, W, C)) or array(T, H, W, C)
    """
    frames = list(frames) if isinstance(frames, np.ndarray) else frames
    clip = moviepy.editor.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, logger=None)
    # skvideo.io.vwrite(out_path, frame_stack)

def dict_to_csv(out_dict, filename):
    with open(filename, 'w') as csv_file:  
        writer = csv.writer(csv_file,delimiter=',')
        out_columns = [[k] + v for k, v in out_dict.items()]
        for row in zip(*out_columns):
            writer.writerow(row)
    return

def debug_mem():
    print("Allocated: ", torch.cuda.memory_allocated(), "Reserved: ", torch.cuda.memory_reserved())

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

def plot_timeseries(timeseries, fig=None, name=None):
    """
    For logging timeseries to wandb tables
    
    Inputs:
    - timeseries: array(T) or list
    
    Returns: plotly graph object figure
    """
    timeseries = np.array(timeseries) if isinstance(timeseries, list) else timeseries
    x = list(range(timeseries.shape[0]))
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=timeseries, mode='lines+markers', name=name))
    fig_html = wandb.Html(fig.to_html(auto_play=False))
    return fig_html

def plot_scatter(x, y, fig=None, title=None, x_title=None, y_title=None):
    """
    For logging scatter plots to wandb tables
    
    Inputs:
    - x: array(T) or list
    - y: array(T) or list
    
    Returns: plt figure
    """
    x = np.array(x) if isinstance(x, list) else x
    y = np.array(y) if isinstance(y, list) else y
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(
            title=dict(text=x_title)
        ),
        yaxis=dict(
            title=dict(text=y_title)
        ),
    )
    fig_html = wandb.Html(fig.to_html(auto_play=False))
    return fig_html

def logging_video(frames, fps=15):
    """
    Prepare videos for logging to wandb tables
    frames should be T,C,H,W
    """
    return wandb.Video(frames, fps=fps)