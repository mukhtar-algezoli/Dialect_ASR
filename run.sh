#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=<mukhtaralgezoli@gmail.com>
#SBATCH --mail-type=ALL

cd ~/projects/def-mageed/mukh/Dialect_ASR
module purge
module load python/3.7.9 scipy-stack
pip install -r requirements.txt
source ~/py37/bin/activate

python main.py 