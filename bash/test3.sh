#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lin00786@umn.edu
#SBATCH -p ram256g

module load python3

source activate torch-env

cd /home/yaoyi/lin00786/weather4cast/weather4cast-preprocessing/

python process_dynamic_variables.py --region R3

