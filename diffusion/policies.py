import torch
import numpy as np
import time
import utils
import clustering
import sc_placement
import legalization

def open_loop(batch_size, model, x_in, cond, intermediate_every = 200, save_videos = False, fps = 60):
    """
    Performs default sampling from diffusion model
    Inputs:
    - x_in: (B, V, F)
    - samples: (B, V, F)

    Note: if save_videos=True, intermediates will contain every video frame
    """
    if save_videos:
        frame_output_rate = 2
    else:
        frame_output_rate = intermediate_every
    samples, intermediates = model.reverse_samples(batch_size, x_in, cond, intermediate_every = frame_output_rate)
    intermediates.append(samples)
    metrics_special = {}
    if save_videos:
        frames = [utils.visualize_placement(intermediate.squeeze(dim=0), cond) for intermediate in intermediates] # T, H, W, C
        # add a couple of frames of the final output at the end
        frames.extend([frames[-1]] * fps)
        log_video = np.moveaxis(np.array(frames), -1, 1) # T, C, H, W
        metrics_special.update({
            "diffusion_video": utils.logging_video(log_video, fps = fps),
            })
    return samples, intermediates, metrics_special

def random(batch_size, x_in, cond):
    """
    Random placements (no consideration for legality etc.) except ports
    Inputs:
    - x_in: (B, V, F)
    Outputs:
    - samples: (B, V, F)
    - intermediates: List[tensor(V, F)]
    - metrics_special: Dict
    """
    B, V, F = x_in.shape
    samples = 2 * torch.rand(x_in.shape, device=x_in.device) - 1
    samples = torch.where(cond.is_ports.view(1, V, 1), x_in, samples)
    return samples

def iterative_clustering(
        batch_size, 
        model, 
        x_in, 
        cond,
        num_iterations = 5,
        num_clusters = 512,
        sc_placement_algorithm = "sgd",
        sc_placement_params = {},
        verbose = False,
        save_plots = False,
        save_videos = False,
        ):
    """
    Performs iterative sampling/reclustering
    Clustering done using kmeans
    Inputs:
    - x_in: (B, V, F)
    - samples: (B, V, F)
    """
    # (1) cluster
    # (2) diffusion
    # (3) uncluster
    # (4) SC placement
    # repeat
    hpwl_clustered_rescaled = []
    hpwl_clustered_normalized = []
    metrics = {}
    metrics_special = {}
    cluster_cond, cluster_x = utils.cluster(cond, num_clusters=num_clusters, placements=x_in, verbose=False)
    # cluster_cond, cluster_x = clustering.cluster(cond, num_clusters=num_clusters, placements=x_in, algorithm="random")

    for i in range(num_iterations):
        debug_plot(cluster_x.squeeze(dim=0), cluster_cond, f"logs/temp/debug_iter_file{cond.file_idx}_iter{i}_clustered.png") if save_plots else 0
        print(f"Finished clustering for sample {cond.file_idx}.") if verbose else 0

        cluster_placement, _ = model.reverse_samples(batch_size, cluster_x, cluster_cond, intermediate_every=0)
        debug_plot(cluster_placement.squeeze(dim=0), cluster_cond, f"logs/temp/debug_iter_file{cond.file_idx}_iter{i}_placed.png") if save_plots else 0
        print(f"Finished diffusion placement for sample {cond.file_idx}.") if verbose else 0
        normalized_hpwl, rescaled_hpwl = utils.hpwl_fast(cluster_placement.squeeze(dim=0), cluster_cond, normalized_hpwl=False)
        hpwl_clustered_rescaled.append(rescaled_hpwl)
        hpwl_clustered_normalized.append(normalized_hpwl)

        samples = utils.uncluster(cluster_cond, cluster_placement)

        # run SC placement
        # TODO see if necessary to place on last step
        if sc_placement_algorithm == "sgd":
            samples = sc_placement.place(
                samples, 
                cond, 
                save_videos = save_videos,
                **sc_placement_params,
                )
        elif sc_placement_algorithm == "adam":
            samples, legalize_metrics, legalize_metrics_special = legalization.legalize_opt(
                samples, 
                cond, 
                save_videos = save_videos,
                **sc_placement_params,
                )
            metrics.update(legalize_metrics)
            metrics_special.update(legalize_metrics_special)
        else:
            raise NotImplementedError
        print(f"Finished SC placement for sample {cond.file_idx}.") if verbose else 0
        debug_plot(samples.squeeze(dim=0), cond, f"logs/temp/debug_iter_file{cond.file_idx}_iter{i}_sc.png") if save_plots else 0


        if i < num_iterations-1: # not on last step; recluster
            # Recluster for next iteration
            cluster_cond, cluster_x = clustering.cluster(cond, num_clusters=num_clusters, placements=samples, algorithm="kmeans")
    
    hpwl_clustered_rescaled = np.array(hpwl_clustered_rescaled)
    hpwl_clustered_normalized = np.array(hpwl_clustered_normalized)
    utils.debug_plot_graph(hpwl_clustered_rescaled, name = f"logs/temp/debug_iter_file{cond.file_idx}_hpwl.png", fig_title = f"lowest:{hpwl_clustered_rescaled.min():.3f}") if save_plots else 0
    # generate and return metrics
    metrics.update({
        "min_clustered_hpwl_rescaled": hpwl_clustered_rescaled.min(),
        "min_clustered_hpwl_normalized": hpwl_clustered_normalized.min(),
    })
    metrics_special.update({
        "iter_hpwl_rescaled": utils.plot_timeseries(hpwl_clustered_rescaled),
        "iter_hpwl_normalized": utils.plot_timeseries(hpwl_clustered_normalized),
    })
    return samples, metrics, metrics_special

def open_loop_clustered(batch_size, model, x_in, cond, intermediate_every = 200):
    # x_in: (B, V, F)
    # samples: (B, V, F)
    t1 = time.time()
    cluster_cond, cluster_x = utils.cluster(cond, num_clusters=512, placements=x_in, verbose=False)
    t2 = time.time()
    cluster_placement, cluster_intermediates = model.reverse_samples(batch_size, cluster_x, cluster_cond, intermediate_every = intermediate_every)
    t3 = time.time()
    
    print("clustering time:", t2-t1, "placing time:", t3-t2)

    samples = utils.uncluster(cluster_cond, cluster_placement)
    intermediates = [utils.uncluster(cluster_cond, intermediate) for intermediate in cluster_intermediates]
    return samples, intermediates

def open_loop_multi(model, x_in, cond, num_attempts, score_fn):
    # x_in: (B, V, F)
    # samples: (B, V, F)
    # generate batch of samples, only return the best according to score_fn
    # assumes 0 score is lowest possible
    B, V, F = x_in.shape
    assert B == 1, "open-loop (multi) policy cannot run in batched mode"

    samples, _ = model.reverse_samples(num_attempts, x_in, cond)
    intermediates = []
    argmax = 0
    max_score = 0
    for i in range(samples.shape[0]):
        score = score_fn(samples[i])
        if score > max_score:
            argmax = i
            max_score = score
        intermediates.append(samples[i:i+1])
    return samples[argmax:argmax+1], intermediates

def iterative(model, x_in, cond, score_fn, num_iter = 4):
    # sort nodes by decreasing size NOTE: experiment with other options for sorting?
    # each reverse sample produces candidate policy
    # going down list of nodes, find first node that produces legality conflict
    # then commit all nodes that do not produce conflict, masking them out
    # repeat reverse sampling for remaining nodes NOTE: experiment with backtracking?
    # finish once all nodes are masked, or iteration limit is reached
    # score_fn: (x, mask) -> bool True if legal
    # samples: (B, V, F)
    B, V, F = x_in.shape
    assert B == 1, "iterative policy cannot run in batched mode"

    # sort vertices by area
    instance_area = compute_instance_area(cond).cpu().numpy()
    instance_order = np.argsort(-instance_area) # index of largest instance first

    masks = [cond.is_ports]
    instances_committed = 0
    intermediates = []
    gen_time = 0
    scoring_time = 0
    samples = x_in
    for iter_idx in range(num_iter):
        t0 = time.time()
        samples, _ = model.reverse_samples(1, samples, cond, mask_override = masks[-1])
        t1 = time.time()
        intermediates.append(samples)
        # update mask
        new_mask = masks[-1].clone()
        scoring_mask = masks[0]|(~masks[-1])
        # commit more instances until fail
        for i in range(instances_committed, V):
            instance = instance_order[i]
            scoring_mask[instance] = False # include next instance in scoring
            if masks[0][instance] or score_fn(samples[0], scoring_mask):
                # commit instance if it's a port or legal enough
                instances_committed += 1
                new_mask[instance] = True # don't move committed instances
            else:
                break
        masks.append(new_mask)
        t2 = time.time()
        gen_time += t1-t0
        scoring_time += t2-t1
        if torch.sum(new_mask) == V:
            break
    commit_rate = (torch.sum(masks[-1])-torch.sum(masks[0])) / (V-torch.sum(masks[0]))
    info = {"gen_t": gen_time, "scoring_t": scoring_time, "commit_rate": commit_rate, "iterations": iter_idx}
    return samples, intermediates, masks[1:], info

def compute_instance_area(cond):
    # cond.x: (V, F)
    # cond.is_ports: (V)
    # get 1D numpy array with sizes of each instance
    areas = cond.x[:,0] * cond.x[:,1]
    return areas

def debug_plot(placement, data, filename = "logs/temp/debug_img.png"):
    img = utils.visualize_placement(placement, data, plot_pins = True, plot_edges = False, img_size=(1024,1024))
    utils.debug_plot_img(img, filename)