# Chip Placement with Diffusion Models

Vint Lee, Minh Nguyen, Leena Elzeiny, Chun Deng, Pieter Abbeel, John Wawrzynek

[Paper](https://arxiv.org/abs/2407.12282)

![Teaser](/media/teaser.png "Diffusion process used to generate placement")

## Installation
Use conda environment found in `environment.yaml`
```
conda env create -f environment.yaml
conda activate chipdiffusion
```

Training and evaluation experiments will log data to Weights & Biases by default. Set your name and W&B project in the [config](diffusion/configs) files using the `logger.wandb_entity` and `logger.wandb_project` options before running the commands below. Turn off W&B logging by appending `logger.wandb=False` to the commands below.

For running evaluations that require clustering, download [shmetis](http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview) and [hmetis](http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview), then place them in the repo's root directory. You may have to run `chmod +x` to allow the programs to run.


## Directory Structure
* [diffusion](diffusion) contains code for training, fine-tuning, and evaluating models
* [data-gen](data-gen) contains code for generating synthetic datasets
* [data-gen/outputs](data-gen/outputs) will be used to store the generated datasets
* [datasets](datasets) is used for other datasets (like IBM and ISPD benchmarks).
* [notebooks](notebooks) has useful scripts and functions for evaluating placements (measuring congestion in particular) and inspecting benchmark files.
* [parsing](parsing) has scripts for converting and clustering benchmarks in the DEF/LEF format (such as IBM).

## Usage

### Data Generation
Generate `v0`, `v1`, and `v2` datasets for training:
```
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v0


PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v1


PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v2 num_train_samples=5000 num_val_samples=2500
```

Configs are also provided for running dataset design experiments. Since we only use these for evaluation, not for training, we only need to generate a few circuits:
```
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=vertex-0.7x num_train_samples=0 num_val_samples=200


PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=distribution-linear num_train_samples=0 num_val_samples=200
```

For experiments on scale factor, the scale factor has to be specified by including `gen_params.edge_dist.dist_params.scale=<SCALE_FACTOR>`. For example:
```
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=scale gen_params.edge_dist.dist_params.scale=0.8 num_train_samples=0 num_val_samples=200
```

For easier debugging, use `data-gen/generate.py`.

### Training Models
After generating data, we train models on the `v1` dataset:

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=train_large task=v1.61
```

We can train smaller models using:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=train_medium task=v1.61 model/size@model.backbone_params=medium


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=train_small task=v1.61 model/size@model.backbone_params=small
```

### Fine-tuning
Once the models have been trained, we can fine-tune them on `v2`:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=finetune_large task=v2.61 mode@_global_=finetune from_checkpoint=v1.61.train_large.61/step_3000000.ckpt
```

### Generating Samples

Evaluating on `v1` dataset without guidance:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py task=v1.61 method=eval from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt legalizer@_global_=none guidance@_global_=none num_output_samples=128
```

Evaluating zero-shot on clustered IBM benchmark with guidance:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm-cluster512 from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt num_output_samples=18
```

Macro-only evaluation for IBM and ISPD benchmarks:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ibm-cluster512 from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt legalizer@_global_=opt-adam num_output_samples=18 model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt legalizer@_global_=opt-adam guidance@_global_=opt num_output_samples=8 model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True
```

Examples of generated placements, for both clustered and macro-only settings, can be found [here](placements).

## Dataset Format
Input netlist is stored using PyTorch-Geometric's [Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch-geometric-data-data) object.

Input placements for training are stored as numpy arrays.

Placement outputs are saved as pickle files containing a single numpy array with (x, y) coordinates for each object.

## Benchmarks
To obtain the IBM dataset, download the benchmark in DEF/LEF format to `benchmarks/ibm` and run the code in [parsing](parsing):
```
PYTHONPATH=. python parsing/cluster.py

PYTHONPATH=. python parsing/cluster.py num_clusters=0
```
The code will parse the DEF/LEFs, cluster the netlists if `num_clusters` is non-zero, then output the dataset as pickle files to `datasets/clustered` directory.

To obtain the ISPD dataset for running evaluations, download the ISPD benchmark in bookshelf format to `benchmarks/ispd2005`, then use [this notebook](notebooks/parse_bookshelf.ipynb).

## Citation
If you found our work useful, please cite:
```
@misc{lee2025chipdiffusion,
      title={Chip Placement with Diffusion Models}, 
      author={Vint Lee and Minh Nguyen and Leena Elzeiny and Chun Deng and Pieter Abbeel and John Wawrzynek},
      year={2025},
      eprint={2407.12282},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.12282}, 
}
```