import utils
import algorithms
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import common
import os
import time
import multiprocessing as mp
from functools import partial

def generate_sample(circuit_gen, base_seed, idx):
    # for use with mp Pools
    torch.manual_seed(base_seed + idx)
    sample = circuit_gen.sample()
    return sample

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cpu'
    out_dir = os.path.join(cfg.out_dir, f"{cfg.dataset_name}.{cfg.seed}")
    checkpointer = common.Checkpointer(os.path.join(out_dir, "latest.ckpt"))
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass
    print(f"saving outputs to: {out_dir}")
    torch.manual_seed(cfg.seed)

    # Generator
    algorithm_table = {
        "v2": algorithms.V2,
        "v3": algorithms.V3,
        "v2_maxsize": algorithms.V2_MaxSize,
        "flora": algorithms.Flora,
    }
    circuit_gen = algorithm_table[cfg.algorithm](**cfg.gen_params)

    # Parallel generator
    parallel_gen = partial(generate_sample, circuit_gen, cfg.seed)

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

    # Start generating
    print(OmegaConf.to_yaml(cfg))
    print(f"==== Start Generation on Device: {device} ====")
    t_0 = time.time()
    t_1 = time.time()

    num_samples = cfg.num_train_samples + cfg.num_val_samples
    while step < num_samples:
        # generate data
        with mp.Pool(processes=cfg.num_workers) as pool:
            sample_batch = pool.map(parallel_gen, range(int(step), int(step) + cfg.print_every), chunksize=1)
        
        step.increment(amount = len(sample_batch))
        t_2 = time.time()
        utils.save_pickle(sample_batch, os.path.join(out_dir, f"{int(step):08d}.pickle"))
        logger.add({
            "time_elapsed": t_2-t_0, 
            "ms_per_step": 1000*(t_2-t_1)/len(sample_batch)
            })
        logger.write()
        checkpointer.save()
        t_1 = t_2
            

if __name__=="__main__":
    main()
