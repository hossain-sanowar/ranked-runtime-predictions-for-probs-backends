#!/bin/bash

#PBS -l select=1:ncpus=24:mem=128GB
#PBS -l place=free
#PBS -l walltime=167:00:00

# restart if job fails
#PBS -r y

#PBS -N main
#PBS -A NeuroB

module load Miniconda/3
source /software/conda/3/etc/profile.d/conda.sh
conda activate base

reg_csv_src_path='/gpfs/project/lebal101/data/regression/all/'
classif_csv_src_path='/gpfs/project/lebal101/data/classification/ranks/'
reg_param_grids_path='/home/lebal101/backend-selection-for-prob-based-on-ranked-runtime-predictions/param_grids/regression/'
classif_param_grids_path='/home/lebal101/backend-selection-for-prob-based-on-ranked-runtime-predictions/param_grids/classification/'
result_path='/gpfs/project/lebal101/results/learners/main'


python3 backend-selection-for-prob-based-on-ranked-runtime-predictions/start_learners.py $reg_csv_src_path $classif_csv_src_path $reg_param_grids_path $classif_param_grids_path $result_path
