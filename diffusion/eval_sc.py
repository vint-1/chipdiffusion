import utils
import sc_placement
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import legalization
import common
import os
import time
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="config_sc")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    sample_input_dir = os.path.join(cfg.log_dir, cfg.from_samples)
    sample_output_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(sample_output_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
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
    # Prepare pre and post processing functions
    preprocess_fns = []
    postprocess_fns = []
    
    def preprocess_fn(x, cond):
        for preprocess_step in preprocess_fns:
            x, cond = preprocess_step(x, cond)
        return x, cond
    def postprocess_fn(x, cond):
        for postprocess_step in postprocess_fns:
            x = postprocess_step(x, cond)
        return x

    # Preparing dataset
    _, target_set = utils.load_graph_data_with_config(cfg.task, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    val_set = []
    # Then uncluster to get validation placements
    samples_set = utils.load_samples(sample_input_dir)
    for (_, target_cond), (samples_x, samples_cond) in zip(target_set, samples_set):
        if cfg.cluster.clustered_samples:
            # uncluster samples
            placement = utils.uncluster(samples_cond, samples_x.unsqueeze(dim=0)).squeeze(dim=0)
        else:
            placement = samples_x
        assert (placement.shape[0] == target_cond.x.shape[0])
        # img = utils.visualize_placement(placement.squeeze(dim=0), target_cond, img_size = (2048, 2048), plot_pins=True)
        # utils.debug_plot_img(img, f"logs/temp/debug_unclustered_{target_cond.file_idx}.png")
        # print("saved image")
        val_set.append((placement, target_cond))

    sample_shape = val_set[0][0].shape
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "guided_diffusion", "skip_diffusion", "skip_guided_diffusion", "no_model"]:
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
            "val_dataset": len(val_set),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists. Here we only load the model
    checkpointer.register({
        "model": model,
    })
    checkpointer.load(
        None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
        else os.path.join(cfg.log_dir, cfg.from_checkpoint)
    )
    
    # Start training
    print(OmegaConf.to_yaml(cfg)) 
    print(f"model has {num_params} params")
    print(f"==== Start Eval on Device: {device} ====")

    # output eval samples
    t3 = time.time()
    print("generating output samples")
    output_metrics = {}
    log_metrics = common.Metrics()
    for i in range(cfg.num_output_samples):
        x, cond = val_set[i]
        metrics, metrics_special, images = sc_placement.save_outputs(
            x, 
            cond,
            save_folder=sample_output_dir,
            device=device, 
            output_number_offset=0,
            sc_placement_algorithm=cfg.eval_policy.sc_placement_algorithm, 
            sc_placement_params=cfg.eval_policy.sc_placement_params, 
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            legalization_fn=legalize_fn,
        )
        print(f"Finished sample {i+1} of {cfg.num_output_samples} \t {metrics}")
        t5 = time.time()
        image_logs = {k: wandb.Image(v) for k, v in images.items()}
        logger.add({
            "reverse_samples": {
                **metrics,
                **metrics_special,
                **image_logs,
                "time_elapsed": t5-t3,
                "num_vertices": x.shape[0],
                "num_edges": cond.edge_index.shape[1],
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
    logger.add(log_metrics.result())
    logger.write()

if __name__=="__main__":
    main()
