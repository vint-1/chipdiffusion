import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import legalization
import analysis_utils
import common
import os
import time
import wandb

def cost(output_metrics):
    """
    Returns dict with cost function(s) for hyperparam sweep
    """
    legality_target = 0.995
    macro_legality_target = 0.998
    legality_temp = 0.001
    hpwl = torch.tensor(output_metrics["hpwl_rescaled"]).mean()
    
    legality = torch.tensor(output_metrics["legality_2"]).mean()
    legality_cost_factor = 1 + 10 * torch.nn.functional.relu((legality_target - legality)/legality_temp)
    
    macro_legality = torch.tensor(output_metrics["macro_legality"]).mean()
    macro_legality_cost_factor = 1 + 10 * torch.nn.functional.relu((macro_legality_target - macro_legality)/legality_temp)
    
    full_cost = (legality_cost_factor * hpwl).item()
    macro_cost = (macro_legality_cost_factor * hpwl).item()
    costs = {
        "cost": full_cost,
        "macro_cost": macro_cost,
    }
    return costs

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)

    # Prepare legalization function
    if cfg.legalization.mode in [None, "none", "None", ""]:
        legalize_fn = None
    elif cfg.legalization.mode == "scheduled":
        def legalize_fn(x, cond):
            return legalization.legalize(
                x, 
                cond,
                **cfg.legalization,
                )
    elif cfg.legalization.mode == "opt":
        def legalize_fn(x, cond):
            return legalization.legalize_opt(
                x, 
                cond,
                **cfg.legalization,
                )
    # Prepare pre and post processing functions. Note that postprocess fns are applied in reverse order
    preprocess_fns = []
    postprocess_fns = []
    if cfg.cluster.is_cluster:
        def cluster_preprocess_fn(x, cond):
            cluster_cond, cluster_x = utils.cluster(cond, cfg.cluster.num_clusters, verbose=cfg.cluster.verbose, placements=x)
            return cluster_x, cluster_cond
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        preprocess_fns.append(cluster_preprocess_fn)
        postprocess_fns.append(cluster_postprocess_fn)
    elif cfg.cluster.cached_clusters:
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        postprocess_fns.append(cluster_postprocess_fn)
    if cfg.sc_halo != 1.0:
        def resize_standard_cells(x, cond):
            _, _, sc_mask = analysis_utils.get_masks(x, cond)
            is_resize = sc_mask.float()
            size_multiplier = (is_resize * cfg.sc_halo) + ((1-is_resize))
            cond.x = cond.x * size_multiplier.unsqueeze(dim=-1)
            return x, cond
        preprocess_fns.append(resize_standard_cells)
    if cfg.edge_dropout > 0.0: # used for debugging
        def edge_dropout(x, cond):
            x, cond = utils.edge_dropout(x, cond, cfg.edge_dropout)
            return x, cond
        preprocess_fns.append(edge_dropout)
    if cfg.macros_only:
        if cfg.cached_macros:
            postprocess_fns.append(utils.add_non_macros)
        else:
            preprocess_fns.append(utils.remove_non_macros)
            postprocess_fns.append(utils.add_non_macros)
    def preprocess_fn(x, cond):
        for preprocess_step in preprocess_fns:
            x, cond = preprocess_step(x, cond)
        return x, cond
    def postprocess_fn(x, cond):
        for i, postprocess_step in enumerate(reversed(postprocess_fns)):
            x, cond = postprocess_step(x, cond)    
        return x, cond

    # Preparing dataset
    train_set, val_set = utils.load_graph_data_with_config(cfg.task, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = val_set[0][0].shape
    dataloader = utils.GraphDataLoader(
        train_set, 
        val_set, 
        cfg.val_batch_size, 
        cfg.val_batch_size, 
        device,
        preprocess_fn = preprocess_fn,
        val_shuffle = False, # Don't shuffle validation set
        )
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "continuous_diffusion", "guided_diffusion", "skip_diffusion", "skip_guided_diffusion", "no_model"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel, 
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
        "no_model": models.NoModel,
    }
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}" if not cfg.param_sweep else None
        wandb_output = common.logger.WandBOutput(wandb_run_name, cfg)
        if cfg.param_sweep:
            with open_dict(cfg):  # for eval/debugging
                cfg.update({
                    "method": f"{cfg.method}.{wandb_output._wandb.run.name}",
                })
        else:
            print("WARNING: param_sweep set to true but wandb disabled. Continuing anyways...")
        outputs.append(wandb_output)
    step = common.Counter()
    logger = common.Logger(step, outputs)

    # Create log and output directories
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"saving checkpoints to: {log_dir}")

    # Output config used
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint if exists. Here we only load the model
    checkpointer.register({
        "model": model,
    })
    checkpointer.load(
        None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
        else os.path.join(cfg.log_dir, cfg.from_checkpoint)
    )

    # Start training
    print(f"model has {num_params} params")
    print(f"==== Start Eval on Device: {device} ====")

    if cfg.eval_samples > 0:
        print("generating evaluation report")
        t1 = time.time()
        utils.generate_report(
            cfg.eval_samples, 
            dataloader, 
            model, 
            logger, 
            policy = cfg.eval_policy_algorithm, 
            intermediate_every = cfg.show_intermediate_every,
            )
        logger.write()
        t2 = time.time()
        print(f"generated report in {t2-t1:.3f} sec")

    # output eval samples
    t3 = time.time()
    print("generating output samples")
    output_metrics = {}
    log_metrics = common.Metrics()
    for i in range(cfg.num_output_samples):
        x, cond = val_set[i]
        metrics, metrics_special, image, image_legalized = utils.save_outputs(
            x, 
            cond, 
            model, 
            save_folder=sample_dir, 
            output_number_offset=0, 
            policy=cfg.eval_policy_algorithm,
            policy_kwargs=cfg.eval_policy,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            legalization_fn=legalize_fn,
        )
        print(f"Finished sample {i+1} of {cfg.num_output_samples} \t {metrics}")
        t5 = time.time()
        # additional metrics
        eig_vals = analysis_utils.get_spectral_info(x, cond, k=1)
        metrics.update({
            "num_vertices": x.shape[0],
            "num_edges": cond.edge_index.shape[1],
            "lambda_2": eig_vals[0],
        })
        logger.add({
            "reverse_samples": {
                **metrics,
                **metrics_special,
                "image": wandb.Image(image_legalized),
                "image_raw": wandb.Image(image),
                "time_elapsed": t5-t3,
            }
        })
        # update metrics
        for k, v in metrics.items():
            if k in output_metrics:
                output_metrics[k].append(v)
            else:
                output_metrics[k] = [v]
        log_metrics.add(metrics)
    utils.dict_to_csv(output_metrics, os.path.join(log_dir,"metrics.csv"))
    for plot_keys in cfg.scatter_plots:
        x_name = plot_keys[0]
        y_name = plot_keys[1]
        if x_name in output_metrics and y_name in output_metrics:
            scatter_plot = utils.plot_scatter(output_metrics[x_name], output_metrics[y_name], x_title=x_name, y_title=y_name)
            logger.add({f"{x_name}_vs_{y_name}": scatter_plot})
    logger.add(log_metrics.result())
    logger.add(cost(output_metrics), prefix = "sweep")
    logger.write()

if __name__=="__main__":
    main()
