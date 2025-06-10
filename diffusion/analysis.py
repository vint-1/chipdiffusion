import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import common
import os
import time
import analysis_utils
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="config_analysis")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    os.makedirs(log_dir, exist_ok = True)
    if cfg.image_limit > 0:
        img_dir = os.path.join(log_dir, "images")
        os.makedirs(img_dir, exist_ok = True)
    print(f"saving results to: {log_dir}")
    torch.manual_seed(cfg.seed)

    # Prepare Clustering pre and post processing
    if cfg.cluster.is_cluster:
        def preprocess_fn(x, cond):
            cluster_cond, cluster_x = utils.cluster(cond, cfg.cluster.num_clusters, verbose=cfg.cluster.verbose, placements=x)
            return cluster_x, cluster_cond
        def postprocess_fn(x, cond):
            return utils.uncluster(cond, x)
    else:
        preprocess_fn = None
        postprocess_fn = None

    # Preparing dataset
    train_set, val_set = utils.load_graph_data_with_config(cfg.task, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    dataloader = utils.GraphDataLoader(
        train_set, 
        val_set, 
        cfg.val_batch_size, 
        cfg.val_batch_size, 
        device,
        preprocess_fn = preprocess_fn,
        val_shuffle = False, # Don't shuffle validation set
        )

    # Prepare logger
    data_metrics = common.Metrics()
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
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
    
    # Start training
    print(OmegaConf.to_yaml(cfg))
    print(f"==== Start Dataset analysis on Device: {device} ====")
    
    t_0 = time.time()
    t_1 = time.time()
    combined_set = train_set + val_set
    for i, (x, cond) in enumerate(combined_set):
        x = x.to(device = device)
        cond.to(device = device)
        if preprocess_fn is not None:
            x, cond = preprocess_fn(x.unsqueeze(dim=0), cond)
        x = x.squeeze(dim=0)
        results, metrics_special = analysis_utils.analyze_sample(x, cond)
        data_metrics.add(results)

        if i < cfg.image_limit:
            img_filename = os.path.join(img_dir, f"{i}.png")
            img = utils.visualize_placement(x, cond, plot_pins = True, plot_edges = cfg.show_edges, img_size=(1024,1024))
            if cfg.table_limit == 0: # no wandb table, save images locally instead
                utils.debug_plot_img(img, img_filename)
                idx = cond.file_idx if "file_idx" in cond else i
                print(f"saved image {img_filename} for sample {idx}.")
            metrics_special["image"] = wandb.Image(img)

        if i < cfg.table_limit:
            logger.add({
                "reverse_samples": metrics_special,
            })

        cond.to(device = "cpu")
        step.increment()

        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            logger.write()
            t_1 = time.time()
    collections = data_metrics.get_all_as_collections()
    analysis_utils.generate_histograms(collections, log_dir)
    analysis_utils.generate_scatterplots(collections, cfg.scatter_plots, log_dir=log_dir, logger=logger)
    
    logger.add(data_metrics.result())
    logger.write()

if __name__=="__main__":
    main()
