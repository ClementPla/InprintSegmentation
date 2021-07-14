#!/usr/bin/env bash

#
#SBATCH --job-name=InprintSegmentation
#SBATCH --output=log_task.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clement.playout@polymtl.ca
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --partition=gpu

ssh -N -L localhost:5010:localhost:5010 clement@m3202-10.demdgi.polymtl.ca & pid=$!

#Load le module anaconda
#source /etc/profile.d/modules.sh
module load anaconda3

source activate ~/.conda/envs/torch18

python main.py

kill $pid