<h1 align='center'>Nequix</h1>

Source code for training [Nequix foundation models](https://arxiv.org/abs/2508.16067) and [Phonon fine-tuning (PFT)](https://arxiv.org/abs/2601.07742).

Model | Dataset | Theory | Reference
---   | ---     | ---    | ---
`nequix-mp-1`| MPtrj | DFT (PBE+U) | [Nequix](https://arxiv.org/abs/2508.16067)
`nequix-mp-1-pft`| MPtrj, MDR Phonon | DFT (PBE+U) |[PFT](https://arxiv.org/abs/2601.07742)
`nequix-omat-1`| OMat24 | DFT (PBE+U, VASP 54) | [PFT](https://arxiv.org/abs/2601.07742)
`nequix-oam-1`| OMat24, sAlex, MPtrj | DFT (PBE+U) | [PFT](https://arxiv.org/abs/2601.07742)
`nequix-oam-1-pft`| OMat24, sAlex, MPtrj, MDR Phonon | DFT (PBE+U) | [PFT](https://arxiv.org/abs/2601.07742)


## Usage

### Installation

```bash
pip install nequix
```

to use [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) kernels,

```bash
pip install nequix[oeq]
# needs to be run after installation:
pip install openequivariance_extjax==0.6.4 --no-build-isolation
```

or for torch (also with kernels):

```bash
pip install nequix[torch]
```

### ASE calculator

Using `nequix.calculator.NequixCalculator`, you can perform calculations in
ASE with a checkpoint produced by the current training code.

```python
from nequix.calculator import NequixCalculator

atoms = ...
atoms.calc = NequixCalculator("checkpoints/model.nqx", backend="jax")
```

or if you want to use the torch backend:

```python
...
atoms.calc = NequixCalculator("checkpoints/model.nqx", backend="torch")
...
```

These are typically comparable in speed with kernels.

Analytical Hessians can be calculated with (currently only supported for JAX backend):

```python
calc = NequixCalculator("checkpoints/model.nqx", backend="jax")
calc.get_hessian(atoms)  # np array of shape (n, n, 3, 3)
```

#### NequixCalculator

Arguments
- `model_path` (str | Path): Path to a current-format local `.nqx` or `.pt` checkpoint.
- `backend` ({"jax", "torch"}, default "jax"): Compute backend.
- `capacity_multiplier` (float, default 1.1): JAX-only; padding factor to limit recompiles.
- `use_compile` (bool, default False): Torch-only; on GPU, uses `torch.compile()`.
- `use_kernel` (bool, default True): on GPU, use [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) kernels.

### Training

Training configs are Python dataclasses registered by name in `nequix/config/runs`.
The `train` command selects either standard JAX training or PFT from the config type:

```bash
uv run train nequix-mp-1
```

Run `uv run train --help` to list every available config name. New runs can reuse an
existing recipe with `dataclasses.replace`, so shared model and dataset settings stay
in one place without YAML inheritance or path handling.

Training and validation data is read exclusively from AtomPack `.atp` files. The
training subset is controlled with `train_frac` and sampled deterministically using
`seed`; the same seed also controls fresh model initialization and epoch shuffling.
When `dataset_name` is set, W&B names include the data schedule. For example,
`dataset_name="1m"`, `train_frac=0.25`, `n_epochs=4`, and
`run_name="nequix_orig"` produce the W&B run name `1m25_4ep_nequix_orig`. Set
`wandb_run_name` to override the generated name.

At successful completion, training prints a two-line CSV summary (one header and one
data row) as the final output. It includes final validation metrics, best validation
loss, run and W&B identifiers, dataset/model/batch sizes, hardware, parameter count,
training and validation time, compute cost in accelerator-hours, and the complete run
configuration. PFT summaries also include the extra validation metrics.

For JAX training with kernels:

```bash
uv sync --extra oeq
uv pip install openequivariance_extjax==0.6.4 --no-build-isolation
uv run train nequix-mp-1
```

All shipped training configs, including OMat and OAM, use JAX. JAX automatically
uses all visible devices, so these runs do not use `torchrun`:

```bash
uv run train nequix-omat-1
uv run train nequix-oam-1
```

To reproduce the training of Nequix-MP-1, first clone the repo and sync the environment:

```bash
git clone https://github.com/atomicarchitects/nequix.git
cd nequix
uv sync
```


Then download the MPtrj data from
https://figshare.com/files/43302033 into `data/` then run the following to extract the data:

```bash
bash data/download_mptrj.sh
```

Preprocess the data into an AtomPack file:

```bash
uv run scripts/preprocess_data.py data/mptrj-gga-ggapu data/mptrj.atp
```

Then start the training run:

```bash
uv run train nequix-mp-1
```

This will take less than 125 hours on a single 4 x A100 node (<25 hours with kernels).
The configured `batch_size` is per device, so you can run on any number of GPUs
(although hyperparameters like learning rate are often sensitive to the resulting
global batch size).

### OMat-1M model scaling study

The depth/width scaling scripts use `data/omat-1m/train.atp` and
`data/omat-1m/val.atp`. They train a 4 x 4 grid of interaction depths and proportional
hidden-irrep widths for four epochs, with one isolated run per GPU. Each run writes a
resumable state, checkpoint, log, and CSV summary locally and logs to the
`nequix-scaling-omat1m` W&B project by default.

Install the OpenEquivariance JAX extension using the commands in the training section
above before launching the default kernel-enabled sweep. For a slower portable run,
pass `--no-kernel`; do not mix kernel and non-kernel trials within one study because
their compute measurements are not comparable.

Preview GPU assignments and commands without starting training:

```bash
uv run python scripts/run_scaling_sweep.py --gpus 0,1,2,3,4,5,6,7 --dry-run
```

Run the seed-0 grid, select the compute/force-MAE Pareto endpoints and knee, then run
two additional seeds for those three finalists:

```bash
uv run python scripts/run_scaling_sweep.py --gpus 0,1,2,3,4,5,6,7
```

The command is safe to rerun: completed trial summaries are skipped and incomplete
trials resume from their epoch state. Use `--phase grid` or `--phase finalists` to run
only one stage, `--max-parallel` to limit concurrent GPUs, and `--wandb-mode offline`
or `disabled` when needed. Results default to `scaling_runs/omat1m-depth-width/`.

Regenerate the aggregate CSV files, scaling fits, plots, and recommendation report
without launching training:

```bash
uv run python scripts/analyze_scaling.py scaling_runs/omat1m-depth-width
```

## Phonon fine-tuning (PFT)


First sync extra dependencies with

```bash
uv sync --extra pft
```

### Phonon calculations

PFT checkpoints produced by the recipes below can be used for both finite-displacement
phonon calculations and analytical Hessians. See
[nequix-examples/phonon](https://github.com/teddykoker/nequix-examples/blob/main/phonon)
for examples.


### Training

Data for the PBE MDR phonon database was originally downloaded and preprocessed with:

```bash
bash data/download_pbe_mdr.sh
uv run data/split_pbe_mdr.py
uv run scripts/preprocess_data_phonopy.py data/mdr-pbe/train data/pbe-mdr/train.atp
uv run scripts/preprocess_data_phonopy.py data/mdr-pbe/val data/pbe-mdr/val.atp
```

To run PFT without co-training run:

```bash
uv run train nequix-mp-1-pft-no-cotrain
```

To run PFT *with* co-training (this also requires `data/mptrj.atp`):

```bash
uv run train nequix-mp-1-pft
```

To run PFT on the OAM base model, follow the data download instructions below and then run:

```bash
uv run train nequix-oam-1-pft
```

Both PFT training runs take about 140 hours on a single A100. Note that PFT training is only currently only supported with the JAX backend, which is both significantly faster and supported by the kernels. See [nequix-examples/pft](https://github.com/teddykoker/nequix-examples/blob/main/pft), which contains a small demo for PFT in PyTorch that can be adapted to other models. Feel free to reach out with questions.

## Training OMat/OAM base models

To reproduce our training runs for the OMat and OAM base models, prepare the
following AtomPack files:

```text
data/omat/train.atp
data/omat/val.atp
data/salex/train.atp
data/salex/val.atp
data/mptrj.atp
```

To train the OMat model, run:
```bash
uv run train nequix-omat-1
```

The OMat recipes run validation every 10,000 optimizer steps within each epoch,
as well as at the end of every epoch. Set `val_every_steps` to a different
positive interval (or `None` for epoch-end-only validation) in a derived config.

For the two-stage OMat foundation-model curriculum (two direct-force epochs,
then two conservative-force epochs), run:

```bash
./scripts/train_omat_foundation_curriculum.sh
```

Both stages use all of `data/omat/train.atp`. Each has an independent resumable
training-state checkpoint under `checkpoints/`. The second stage initializes from
the best first-stage backbone checkpoint but deliberately creates a fresh optimizer
and learning-rate schedule for the new objective. The final best checkpoint is
`checkpoints/nequix-omat-foundation-conservative.nqx`.

The OMat run writes its best checkpoint to `checkpoints/nequix-omat-1.nqx`.
To fine-tune the OAM model from that newly trained checkpoint, run
```bash
uv run train nequix-oam-1
```
## Citation

```bibtex
@article{koker2026pft,
  title={{PFT}: Phonon Fine-tuning for Machine Learned Interatomic Potentials},
  author={Koker, Teddy and Gangan, Abhijeet and Kotak, Mit and Marian, Jaime and Smidt, Tess},
  journal={arXiv preprint arXiv:2601.07742},
  year={2026}
}

@article{koker2025training,
  title={Training a foundation model for materials on a budget},
  author={Koker, Teddy and Kotak, Mit and Smidt, Tess},
  journal={arXiv preprint arXiv:2508.16067},
  year={2025}
}
```
