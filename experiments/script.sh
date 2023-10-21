#!/bin/bash

#SBATCH -N 1
#SBATCH -t 1-00:00                      # wall time (D-HH:MM)
#SBATCH -p publicgpu                      # Use gpu partition
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -n 1

#SBATCH -o mars-crater/logs/u2netCCE_multiclass_og.%j.out                       # STDOUT (%j = JobId)
#SBATCH -e mars-crater/logs/u2netCCE_multiclass_og.%j.err                       # STDERR (%j = JobId)

##SBATCH -A smalvi                    # Account hours will be pulled from (commented out with double # in front)
##SBATCH --mail-type=ALL                 # Send a notification when the job starts, stops, or fails
##SBATCH --mail-user=smalvi@asu.edu    # send-to address

#module purge                           # Always purge modules to ensure a consistent environment

#conda activate mars-crater

cd mars-crater/

module load anaconda/py3
source activate mars-crater-gpu2
python train.py
source deactivate mars-crater-gpu2