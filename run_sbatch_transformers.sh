#!/bin/bash -l
#SBATCH --job-name=pluramath_transformers
#SBATCH --output=./logs/pluramath_transformers_%j.log
#SBATCH --error=./logs/pluramath_transformers_%j.log
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Uncomment and edit if your cluster requires an HTTP(S) proxy.
# export HTTP_PROXY=http://proxy.example.edu:80
# export HTTPS_PROXY=http://proxy.example.edu:80
# export http_proxy=http://proxy.example.edu:80
# export https_proxy=http://proxy.example.edu:80

# Do not use packages from ~/.local/lib/python...
# Use only packages installed inside the active conda env.
export PYTHONNOUSERSITE=1

echo "Job started on:"
hostname
date

echo "Slurm job id:"
echo "$SLURM_JOB_ID"

echo "CUDA from nvidia-smi:"
nvidia-smi

source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-nlp}"

# echo "Python:"
# which python
# python --version
# python -c "import sys; print(sys.executable)"

# echo "Torch CUDA check:"
# python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('cuda version:', torch.version.cuda)"

# echo "Transformers check:"
# python -c "import transformers, packaging; print('transformers:', transformers.__version__); print('packaging ok')"

echo "Running Python script..."
BATCH_SIZE=10 ./run_main_experiment_transformers_foreground.sh LiquidAI/LFM2.5-1.2B-Thinking LFM2.5-1.2B-Thinking
