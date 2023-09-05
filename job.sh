#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --output=op_file.txt
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
module load cuda/10.0
module add cuda/10.0


export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /scratch/
if [ ! -d aman.atman ]; then
    mkdir aman.atman
fi

cd aman.atman/
rm -rf csdi
if [ ! -d csdi ]; then  
    mkdir csdi
fi

cd csdi/
# rm -r *
rsync -avz aman.atman@ada.iiit.ac.in:/home2/aman.atman/csdi/ --exclude=runs --exclude=.git/ --exclude=.mypy_cache/  --exclude=runs --exclude=log --exclude=logs --exclude=__pycache__  --exclude=.git/  --exclude=wandb --exclude=.mypy  --exclude=bgcn/data/processed_dir --exclude=wandb --exclude=results/ ./


# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate g
export WANDB_API_KEY=e0aef73e93ca47982d728049617662074f43d505
python exe_pems.py

# rsync to original location
# rsync -avz . --exclude=runs --exclude=.git/  --exclude=wandb  aman.atman@ada.iiit.ac.in:/home2/aman.atman/grin/
