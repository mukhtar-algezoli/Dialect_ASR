#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=3:00
#SBATCH --account=def-mageed
#SBATCH --mail-user=<mukhtaralgezoli@gmail.com>
#SBATCH --mail-type=ALL
nvidia-smi

cd ~/projects/def-mageed/mukh/Dialect_ASR
module purge
module load python/3.7.9 scipy-stack
pip install -r requirements.txt
source ~/py37/bin/activate

python main.py 