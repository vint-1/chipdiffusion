import torch
import guidance
import utils
import numpy as np

def legalize(
        x, 
        cond, 
        step_size, 
        grad_descent_steps, 
        softmax_min, 
        softmax_max, 
        save_videos = False,
        legality_weight = 1.0, 
        hpwl_weight = 0.0,
        softmax_critical_factor = 3/4,
        guidance_critical_factor = 3/4,
        zero_hpwl_factor = 9/10,
        legality_increase_factor = 2.0,
        macros_only = False,
        **kwargs,
        ):
    """
    Perform legalization using gradient descent
    Inputs:
    x - tensor(B, V, 2) normalized 2D coordinates on CUDA (if possible)
    cond - Data on CUDA (if possible)
    """
    mask = get_legalization_mask(cond, macros_only=macros_only)

    x_current = x.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD((x_current,), lr=step_size, momentum=0.0)

    softmax_critical_step = round(grad_descent_steps * softmax_critical_factor)
    guidance_critical_step = round(grad_descent_steps * guidance_critical_factor)
    zero_hpwl_step = round(grad_descent_steps * zero_hpwl_factor)
    softmax_factors = linear_schedule(grad_descent_steps, 0, softmax_critical_step, softmax_min, softmax_max)
    legality_schedule = linear_schedule(grad_descent_steps, guidance_critical_step, grad_descent_steps, legality_weight, legality_weight * legality_increase_factor)
    hpwl_schedule = linear_schedule(grad_descent_steps, guidance_critical_step, zero_hpwl_step, hpwl_weight, 0)
    
    video_frames = []
    legalities = []
    hpwls_normalized = []
    hpwls_rescaled = []
    metrics = {}
    metrics_special = {}

    # m step gradient descent
    with torch.enable_grad():
        for i, softmax_factor in zip(range(grad_descent_steps), softmax_factors):
            optimizer.zero_grad()
            h_legality = legality_schedule[i] * guidance.legality_guidance_potential_tiled(x_current, cond, softmax_factor=softmax_factor, mask=mask)
            # we have to manually rescale because the tiled implementation already called backward()
            assert (not h_legality.requires_grad), "tiled version of h_legality assumed to not have grad"
            x_current.grad *= legality_schedule[i]

            if hpwl_weight != 0.0:
                h_hpwl = hpwl_schedule[i] * guidance.hpwl_guidance_potential(x_current, cond)
            else:
                h_hpwl = torch.zeros_like(h_legality).requires_grad_(True)
            h = h_legality + h_hpwl
            h.sum().backward()
            # Stop ports from moving due to hpwl guidance
            x_current.grad *= (~mask).float()
            optimizer.step()

            if save_videos and (i+1) % 50 == 0:
                # make video and measure hpwl/legality
                x_evaluate = x_current[0].detach()
                image = utils.visualize_placement(x_evaluate, cond, img_size = (512, 512))
                video_frames.append(image)

                hpwl_normalized, hpwl_rescaled = utils.hpwl_fast(x_evaluate, cond, normalized_hpwl=False)
                legality = utils.check_legality_new(x_evaluate, x_evaluate, cond, cond.is_ports, score=True)
                hpwls_normalized.append(hpwl_normalized)
                hpwls_rescaled.append(hpwl_rescaled)
                legalities.append(legality)

    if len(video_frames) > 0:
        # TODO don't save to disk
        utils.save_video(video_frames, f"logs/temp/debug_legalization_{cond.file_idx}.mp4", fps = 20)
        log_video = np.moveaxis(np.array(video_frames), -1, 1) # T, C, H, W
        metrics_special.update(
            {
                "legalization_video": utils.logging_video(log_video, fps = 20), 
                "legalization_hpwl_rescaled": utils.plot_timeseries(hpwls_rescaled),
                "legalization_hpwl_normalized": utils.plot_timeseries(hpwls_normalized),
                "legalization_legality": utils.plot_timeseries(legalities),
            }
        )
    
    return x_current.detach(), metrics, metrics_special

def legalize_opt(
        x, 
        cond, 
        step_size, 
        grad_descent_steps, 
        softmax_min, 
        softmax_max, 
        save_videos = False,
        save_timeseries = False, 
        softmax_critical_factor = 1.0,
        hpwl_weight = 2e-5, # This should be static
        alpha_init = 1.0,
        alpha_lr = 5e-2,
        legality_potential_target = 1e-4,
        use_adam = False,
        macros_only = False,
        **kwargs,
        ):
    """
    Perform legalization using gradient descent
    Auto-tunes legality weight using optimization principles (inspired by SAC)
    Inputs:
    x - tensor(B, V, 2) normalized 2D coordinates on CUDA (if possible)
    cond - Data on CUDA (if possible)
    """
    assert hpwl_weight > 0.0, "Don't use this if hpwl_weight is 0"
    mask = get_legalization_mask(cond, macros_only=macros_only)

    x_current = x.detach().clone().requires_grad_(True)
    alpha = torch.tensor(alpha_init, dtype = x_current.dtype, device = x_current.device, requires_grad = True)
    if use_adam:
        optimizer_x = torch.optim.Adam((x_current,), lr=step_size, betas=(0.8, 0.99))
        optimizer_alpha = torch.optim.Adam((alpha,), lr=alpha_lr, betas=(0.9, 0.99))
    else:
        optimizer_x = torch.optim.SGD((x_current,), lr=step_size, momentum=0.0)
        optimizer_alpha = torch.optim.SGD((alpha,), lr=alpha_lr, momentum=0.0)

    softmax_critical_step = round(grad_descent_steps * softmax_critical_factor)
    softmax_factors = linear_schedule(grad_descent_steps, 0, softmax_critical_step, softmax_min, softmax_max)
    
    video_frames = []
    alphas = []
    legality_potentials = []
    legalities = []
    hpwls_normalized = []
    hpwls_rescaled = []
    metrics = {}
    metrics_special = {}

    # m step gradient descent
    with torch.enable_grad():
        for i, softmax_factor in zip(range(grad_descent_steps), softmax_factors):
            # gradient step wrt x
            optimizer_x.zero_grad()
            legality_raw_potential = guidance.legality_guidance_potential_tiled(x_current, cond, softmax_factor=softmax_factor, mask=mask)
            h_legality = alpha.detach().item() * legality_raw_potential
            # we have to manually rescale because the tiled implementation already called backward()
            assert (not h_legality.requires_grad), "tiled version of h_legality assumed to not have grad"
            x_current.grad *= alpha.detach().item()
            
            h_hpwl = hpwl_weight * guidance.hpwl_guidance_potential(x_current, cond)
            h = h_legality + h_hpwl
            h.sum().backward()
            # Stop ports from moving due to hpwl guidance
            x_current.grad *= (~mask).float()
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

            if (i+1) % 50 == 0:
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

    if len(video_frames) > 0:
        log_video = np.moveaxis(np.array(video_frames), -1, 1) # T, C, H, W
        metrics_special.update(
            {
                "legalization_video": utils.logging_video(log_video, fps = 20),
            }
        )
    if save_timeseries:
        metrics_special.update(
            { 
                "legalization_hpwl_rescaled": utils.plot_timeseries(hpwls_rescaled),
                "legalization_hpwl_normalized": utils.plot_timeseries(hpwls_normalized),
                "legalization_legality": utils.plot_timeseries(legalities),
            }
        )
    metrics_special.update(
        {
            "legalization_alphas": utils.plot_timeseries(alphas),
            "legalization_potentials": utils.plot_timeseries(legality_potentials),
        }
    )
    return x_current.detach(), metrics, metrics_special

def get_legalization_mask(cond, macros_only=False):
    if macros_only:
        return (~get_mask(cond, mask_key="is_macros")) | get_mask(cond)
    else:
        return get_mask(cond)

def get_mask(cond, mask_key="is_ports"):
    if mask_key and mask_key in cond:
        mask = cond[mask_key]
        V = cond.num_nodes
        mask = mask.view(1, V, 1)
        return mask
    else:
        return None

def linear_schedule(total_steps, start_idx, end_idx, start_val, end_val):
    """
    static at start_val until start_idx, then linearly interpolates to end_val
    """
    schedule = torch.full((total_steps,), fill_value = start_val, dtype=torch.float32)
    schedule[start_idx:end_idx] = torch.linspace(start_val, end_val, end_idx-start_idx)
    schedule[end_idx:] = end_val
    return schedule