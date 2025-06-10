import torch
import guidance
import utils
import time
import os
import pickle
import numpy as np
from common.timer import Timer

@torch.no_grad()
def save_outputs(
        x_in, 
        cond,
        save_folder,
        device, 
        output_number_offset=0, 
        sc_placement_algorithm="sgd", 
        sc_placement_params = {}, 
        preprocess_fn=None, 
        postprocess_fn=None, 
        legalization_fn=None,
        ):

    """
    Performs SC placement and saves results to disk

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
    x_in = torch.unsqueeze(x_in, dim=0).to(device)
    original_device = cond.x.device
    cond.to(device)
    metrics = {}
    metrics_special = {} # For things that should not be aggregated like plots, images, etc.

    # user-defined preprocess function
    t0 = time.time()
    x_preprocessed, cond_preprocessed = preprocess_fn(x_in, cond) if preprocess_fn is not None else (x_in, cond)

    image_unplaced = utils.visualize_placement(x_preprocessed[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))

    t1 = time.time()
    if sc_placement_algorithm == "sgd":
        sample = place(x_preprocessed, cond_preprocessed, **sc_placement_params)
    elif sc_placement_algorithm == "adam_2stage":
        sample, placement_metrics, placement_metrics_special = place_2stage(x_preprocessed, cond_preprocessed, **sc_placement_params)
        metrics.update(placement_metrics)
        metrics_special.update(placement_metrics_special)
    else:
        raise NotImplementedError
    t2 = time.time()

    # save image too
    image = utils.visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))

    # legalization
    if legalization_fn is not None:
        sample, legalization_metrics, legalization_metrics_special = legalization_fn(sample, cond_preprocessed)
        metrics.update(legalization_metrics)
        metrics_special.update(legalization_metrics_special)
        image_legalized = utils.visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
    else:
        image_legalized = image
    utils.debug_plot_img(image_legalized, os.path.join(save_folder, f"placed{cond.file_idx}"))

    # user-defined postprocess function
    sample_unprocessed = sample.detach().clone()
    sample = postprocess_fn(sample, cond_preprocessed) if postprocess_fn is not None else sample

    sample = sample.squeeze(dim=0).detach().to(device = cond.x.device)
    sample = utils.postprocess_placement(sample, cond).cpu().numpy() # mandatory post-processing
    idx = cond.file_idx if "file_idx" in cond else output_number_offset
    save_file = os.path.join(save_folder, f"sample{idx}.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(sample, f)
    t3 = time.time()
    
    # evaluate sample and generate sampling metrics
    hpwl_normalized, hpwl_rescaled = utils.hpwl_fast(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    legality = utils.check_legality_new(sample_unprocessed[0], x_in[0], cond_preprocessed, cond_preprocessed.is_ports, score=True)
    t4 = time.time()

    cond.to(original_device)

    metrics.update({
        "idx": cond.file_idx,
        "hpwl_normalized": hpwl_normalized,
        "hpwl_rescaled": hpwl_rescaled,
        "legality_2": legality,
        "model_time": t2-t1,
        "generation_time": t3-t0,
        "eval_time": t4-t3,
    })
    images = {
        "image": image_legalized,
        "image_raw": image,
        "image_unplaced": image_unplaced,
    }
    return metrics, metrics_special, images

def place(
        x, 
        cond, 
        lr, 
        grad_descent_steps, 
        softmax_min, 
        softmax_max, 
        legality_guidance_weight = 0.0, 
        hpwl_guidance_weight = 1.0,
        save_videos = True,
        ):
    """
    Perform SC placement using gradient descent
    Inputs:
    x - tensor(B, V, 2) normalized 2D coordinates on CUDA (if possible)
    cond - Data on CUDA (if possible)
    """
    is_ports = get_mask(x, cond, mask_key="is_ports")
    is_macros = get_mask(x, cond, mask_key="is_macros")
    legality_mask = is_macros | is_ports # masks out ports AND macros from legality guidance
    movement_mask = (~(is_macros | is_ports)) # True if object is moveable

    x_current = x.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD((x_current,), lr=lr)

    softmax_factors = torch.linspace(softmax_min, softmax_max, grad_descent_steps)
    legality_schedule = linear_schedule(grad_descent_steps, grad_descent_steps//2, grad_descent_steps, 0.0, legality_guidance_weight)
    
    video_frames = []
    h_values = []
    # m step gradient descent
    with torch.enable_grad():
        for i, softmax_factor, legality_weight in zip(range(grad_descent_steps), softmax_factors, legality_schedule):
            optimizer.zero_grad()
            
            if legality_guidance_weight != 0.0:
                h_legality = legality_weight * guidance.legality_guidance_potential(x_current, cond, softmax_factor=softmax_factor, mask=legality_mask)
            else:
                h_legality = 0
            
            if hpwl_guidance_weight != 0.0:
                h_hpwl = hpwl_guidance_weight * guidance.hpwl_guidance_potential(x_current, cond)
            else:
                h_hpwl = 0
            
            h = h_legality + h_hpwl
            h_sum = h.sum()
            h_sum.backward()
            # mask gradients for x_current so that macros don't move
            x_current.grad *= movement_mask.float()
            optimizer.step()

            if save_videos and (i+1) % 8 == 0:
                image = utils.visualize_placement(x_current.squeeze(dim=0), cond, img_size = (512, 512))
                video_frames.append(image) # H,W,C
                h_values.append(h_sum.detach().cpu().item())
    
    # make video and save to disk
    if len(video_frames) > 0:
        utils.save_video(video_frames, f"logs/temp/debug_scplace_{cond.file_idx}.mp4", fps = 8)
        utils.debug_plot_graph(h_values, name = f"logs/temp/debug_scplace_{cond.file_idx}_h.png")
        print(f"Saved video for {cond.file_idx} SC placement")

    return x_current.detach()

def place_2stage(
        x, 
        cond, 
        step_size, 
        rearrangement_steps,
        optimization_steps, 
        softmax_min, 
        softmax_max,
        softmax_critical_factor = 1.0,
        init_mode = "use_placement", 
        hpwl_weight = 2e-5, # This should be static
        alpha_init = 1.0,
        alpha_lr = 5e-2,
        legality_potential_target = 1e-4,
        legality_include_macros = True, # include macros in legality potential computation
        use_adam = False,
        save_videos = False, # These options take a lot of time
        save_timeseries = False,
        ):
    """
    Perform 2-stage SC placement, with rearrangement stage, followed by optimization stage
    Inputs:
    x - tensor(B, V, 2) normalized 2D coordinates on CUDA (if possible)
    cond - Data on CUDA (if possible)
    """
    is_ports = get_mask(x, cond, mask_key="is_ports")
    is_macros = get_mask(x, cond, mask_key="is_macros")
    if legality_include_macros:
        legality_mask = is_ports
    else:
        legality_mask = is_macros | is_ports # masks out ports AND macros from legality guidance
    movement_mask = (~(is_macros | is_ports)) # True if object is moveable

    x_current = x.detach().clone()
    # initialize SC placement
    if init_mode == "use_placement":
        pass
    elif init_mode == "zero":
        x_current = torch.where(movement_mask, torch.zeros_like(x_current), x_current)
    elif init_mode == "random":
        random_positions = 2*torch.rand_like(x_current)-1
        x_current = torch.where(movement_mask, random_positions, x_current)
    else:
        raise NotImplementedError
    x_current.requires_grad_(True)

    alpha = torch.tensor(alpha_init, dtype = x_current.dtype, device = x_current.device, requires_grad = True)
    if use_adam:
        optimizer_x = torch.optim.Adam((x_current,), lr=step_size, betas=(0.8, 0.99))
        optimizer_alpha = torch.optim.Adam((alpha,), lr=alpha_lr, betas=(0.9, 0.99))
    else:
        optimizer_x = torch.optim.SGD((x_current,), lr=step_size, momentum=0.0)
        optimizer_alpha = torch.optim.SGD((alpha,), lr=alpha_lr, momentum=0.0)

    softmax_critical_step = round(optimization_steps * softmax_critical_factor)
    softmax_factors = linear_schedule(optimization_steps, 0, softmax_critical_step, softmax_min, softmax_max)
    
    video_frames = []
    alphas = []
    legality_potentials = []
    legalities = []
    hpwls_normalized = []
    hpwls_rescaled = []
    metrics = {}
    metrics_special = {}

    rearr_grad_timer = Timer()
    opt_grad_timer = Timer()
    misc_timer = Timer()

    # m step gradient descent
    with torch.enable_grad():
        for i in range(rearrangement_steps):
            rearr_grad_timer.start()
            
            # gradient step wrt x
            optimizer_x.zero_grad()
            h_hpwl = hpwl_weight * guidance.hpwl_guidance_potential(x_current, cond)
            h_hpwl.sum().backward()
            # Stop ports from moving due to hpwl guidance
            x_current.grad *= movement_mask.float()
            optimizer_x.step()

            rearr_grad_timer.stop()

            if (i+1) % 50 == 0:
                misc_timer.start()
                x_evaluate = x_current[0].detach()
                if save_videos:
                    # make video 
                    image = utils.visualize_placement(x_evaluate, cond, img_size = (512, 512))
                    video_frames.append(image)

                if save_timeseries:
                    # measure hpwl/legality
                    hpwl_normalized, hpwl_rescaled = utils.hpwl_fast(x_evaluate, cond, normalized_hpwl=False)
                    legality = utils.check_legality_new(x_evaluate, x_evaluate, cond, cond.is_ports, score=True)
                    hpwls_normalized.append(hpwl_normalized)
                    hpwls_rescaled.append(hpwl_rescaled)
                    legalities.append(legality)
                misc_timer.stop()

        for i, softmax_factor in zip(range(optimization_steps), softmax_factors):
            opt_grad_timer.start()

            # gradient step wrt x
            optimizer_x.zero_grad()
            legality_raw_potential = guidance.legality_guidance_potential_tiled(x_current, cond, softmax_factor=softmax_factor, mask=legality_mask)
            h_legality = alpha.detach().item() * legality_raw_potential
            # we have to manually rescale because the tiled implementation already called backward()
            assert (not h_legality.requires_grad), "tiled version of h_legality assumed to not have grad"
            x_current.grad *= alpha.detach().item()
            h_hpwl = hpwl_weight * guidance.hpwl_guidance_potential(x_current, cond)
            h = h_legality + h_hpwl
            h.sum().backward()
            # Stop ports from moving due to hpwl guidance
            x_current.grad *= movement_mask.float()
            optimizer_x.step()

            # gradient step wrt alpha
            optimizer_alpha.zero_grad()
            alpha_cost = -alpha * (legality_raw_potential.detach() - legality_potential_target)
            alpha_cost.backward()
            optimizer_alpha.step()
            
            # for numerical stability
            alpha.data.clip_(max=15)

            # record optimization metrics at every step
            alphas.append(alpha.detach().cpu().item())
            legality_potentials.append(legality_raw_potential.detach().cpu().item())

            opt_grad_timer.stop()

            if (i+1) % 50 == 0:
                misc_timer.start()

                x_evaluate = x_current[0].detach()
                if save_videos:
                    # make video 
                    image = utils.visualize_placement(x_evaluate, cond, img_size = (512, 512))
                    video_frames.append(image)

                if save_timeseries:
                    # measure hpwl/legality
                    hpwl_normalized, hpwl_rescaled = utils.hpwl_fast(x_evaluate, cond, normalized_hpwl=False)
                    legality = utils.check_legality_new(x_evaluate, x_evaluate, cond, cond.is_ports, score=True)
                    hpwls_normalized.append(hpwl_normalized)
                    hpwls_rescaled.append(hpwl_rescaled)
                    legalities.append(legality)

                misc_timer.stop()

    if len(video_frames) > 0:
        log_video = np.moveaxis(np.array(video_frames), -1, 1) # T, C, H, W
        metrics_special.update({
            "sc_video": utils.logging_video(log_video, fps = 20),
            })
    if save_timeseries:
        metrics_special.update({
                "sc_hpwl_rescaled": utils.plot_timeseries(hpwls_rescaled),
                "sc_hpwl_normalized": utils.plot_timeseries(hpwls_normalized),
                "sc_legality": utils.plot_timeseries(legalities),
                })
    metrics_special.update({
        "sc_alphas": utils.plot_timeseries(alphas),
        "sc_legality_potentials": utils.plot_timeseries(legality_potentials),
        })
    metrics.update({
        "sc_rearrangement_time": rearr_grad_timer.read(),
        "sc_opt_time": opt_grad_timer.read(),
        "sc_misc_time": misc_timer.read(),
    })
    return x_current.detach(), metrics, metrics_special

def get_mask(x, cond, mask_key="is_ports"):
    if mask_key and mask_key in cond: # TODO raise error if mask key unexpectedly missing
        mask = cond[mask_key]
        B, V, F = x.shape
        mask = mask.view(1, V, 1)
        return mask
    else:
        raise KeyError(f"mask key {mask_key} not found in graph Data")
    
def linear_schedule(total_steps, start_idx, end_idx, start_val, end_val):
    """
    static at start_val until start_idx, then linearly interpolates to end_val
    """
    schedule = torch.full((total_steps,), fill_value = start_val, dtype=torch.float32)
    schedule[start_idx:end_idx] = torch.linspace(start_val, end_val, end_idx-start_idx)
    schedule[end_idx:] = end_val
    return schedule