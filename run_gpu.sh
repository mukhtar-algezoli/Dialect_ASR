#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --account=def-mageed
#SBATCH --mem=32G              # Request the full memory of the node

cd ~/projects/def-mageed/mukh/Dialect_ASR
module purge
module load python/3.7.9 scipy-stack
pip install -r requirements.txt
source ~/py37/bin/activate

python main.py 