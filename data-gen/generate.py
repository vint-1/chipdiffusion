import utils
import algorithms
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import common
import os
import time
import common.timer as timer

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(cfg.out_dir, f"{cfg.dataset_name}.{cfg.seed}")
    checkpointer = common.Checkpointer(os.path.join(out_dir, "latest.ckpt"))
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass
    print(f"saving outputs to: {out_dir}")

    # Generator
    algorithm_table = {
        "v2": algorithms.V2,
        "v3": algorithms.V3,
        "v2_maxsize": algorithms.V2_MaxSize,
        "flora": algorithms.Flora,
    }
    circuit_gen = algorithm_table[cfg.algorithm](**cfg.gen_params)

    # Prepare logger
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"circuit_gen.{cfg.dataset_name}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(out_dir, "config.yaml"))
    
    checkpointer.register({
        "step": step,
    })
    checkpointer.load()
    torch.manual_seed(cfg.seed + int(step))

    # Start training
    print(OmegaConf.to_yaml(cfg))
    print(f"==== Start Generation on Device: {device} ====")
    t_0 = time.time()
    t_1 = time.time()
    size_dist_timer = timer.Timer()
    place_timer = timer.Timer()
    terminal_timer = timer.Timer()
    edge_timer = timer.Timer()
    samples = []

    if cfg.debug:
        print("DEBUGGING MODE ENABLED !")
        total_v = 0
        total_e = 0
    num_samples = cfg.num_train_samples + cfg.num_val_samples
    while step < num_samples:
        # generate data
        sample = circuit_gen.sample(
            size_dist_timer=size_dist_timer,
            place_timer=place_timer,
            terminal_timer=terminal_timer,
            edge_timer=edge_timer,
        )
        samples.append(sample)
        # append to samples
        step.increment()
        
        if cfg.debug:
            x, cond = sample
            V = x.shape[0]
            E = cond.edge_index.shape[1]
            total_v += V
            total_e += E
            print(int(step), V, E, total_v/int(step), total_e/int(step))
            img = utils.visualize_placement(x, cond)
            utils.debug_plot_img(img, os.path.join(out_dir, f"{int(step)}sample"))

        if ((int(step)) % cfg.print_every == 0) or (step == num_samples):
            t_2 = time.time()
            # TODO use something better than pickle
            utils.save_pickle(samples, os.path.join(out_dir, f"{int(step):08d}.pickle"))
            samples = []
            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every,
                "size_dist_time": size_dist_timer.reset(),
                "place_time": place_timer.reset(),
                "terminal_time": terminal_timer.reset(),
                "edge_time": edge_timer.reset(),
                })
            logger.write()
            checkpointer.save()
            t_1 = t_2
            

if __name__=="__main__":
    main()
