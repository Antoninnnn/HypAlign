---
description: Generate and submit a SLURM job to TAMU HPRC (Grace cluster, A100 nodes)
---

Generate a SLURM batch script and submit it to TAMU HPRC Grace cluster.
Ask the user which job to submit if not specified.

## Available jobs

| Job ID | Script | Model | Est. wall time |
|--------|--------|-------|---------------|
| `probe-650m` | `run_experiment_go.py` | ESM-2 650M frozen | 4 h |
| `finetune-150m` | `run_finetune_go.py` | ESM-2 150M FT | 6 h |
| `finetune-650m` | `run_finetune_go.py` | ESM-2 650M FT | 12 h |

## Steps

### 1. Verify data is on cluster

```bash
ls $SCRATCH/hyp_ssf_probe/cache/protst_go_mf_decoded.pt
```
If missing, rsync from local:
```bash
# Run this on LOCAL machine, not cluster
rsync -avz experiments/hyp_ssf_probe/cache/ \
    netid@grace.hprc.tamu.edu:$SCRATCH/hyp_ssf_probe/cache/
rsync -avz experiments/hyp_ssf_probe/*.py \
    netid@grace.hprc.tamu.edu:$SCRATCH/hyp_ssf_probe/
```

### 2. Write SLURM script

Create `submit_<job_id>.sh` in the experiment directory. Template:

```bash
#!/bin/bash
#SBATCH --job-name=hyp-ssf-<job_id>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=<wall_time>:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yining_yang@tamu.edu

module purge
module load GCC/12.2.0 CUDA/12.4.0 Miniconda3/23.5.2-0

conda activate pannot-infer

cd $SCRATCH/hyp_ssf_probe
python -u <script.py>
```

**Job-specific substitutions:**

`probe-650m`:
- wall_time: `4`
- script.py: `run_experiment_go.py`
- Before running: edit `ESM2_HF = "facebook/esm2_t33_650M_UR50D"` and
  `ESM_DIM = 1280` in `run_experiment_go.py`

`finetune-150m`:
- wall_time: `8`
- script.py: `run_finetune_go.py`
- Config already set for 150M

`finetune-650m`:
- wall_time: `12`
- script.py: `run_finetune_go.py`
- Before running: set `ESM_MODEL = "facebook/esm2_t33_650M_UR50D"`,
  `BATCH_SIZE = 8`, `GRAD_ACCUM = 8` in `run_finetune_go.py`

### 3. Submit

```bash
mkdir -p logs
sbatch submit_<job_id>.sh
squeue -u $USER    # monitor
```

## Environment setup (one-time on HPRC)

```bash
module load GCC/12.2.0 CUDA/12.4.0 Miniconda3/23.5.2-0
conda create -n pannot-infer python=3.10 -y
conda activate pannot-infer
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate scikit-learn scipy umap-learn matplotlib
```

## Retrieving results

After job completes, rsync results back:
```bash
rsync -avz netid@grace.hprc.tamu.edu:$SCRATCH/hyp_ssf_probe/checkpoints/ \
    experiments/hyp_ssf_probe/checkpoints/
rsync -avz netid@grace.hprc.tamu.edu:$SCRATCH/hyp_ssf_probe/results/ \
    experiments/hyp_ssf_probe/results/
```
Then run `/evaluate` locally to generate analysis figures.
