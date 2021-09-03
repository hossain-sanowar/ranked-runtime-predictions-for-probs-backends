# Ranking Model Checking Backends for Automated Selection

This repository contains the code for the experiments conducted on our paper,
"Ranking Model Checking Backends for Automated Selection via Classification and Regression Learning" by Dunkelau & Baldus (2021),
to appear in the proceedings of the [OVERLAY'21 Workshop](https://overlay.uniud.it/workshop/2021/).

## Repo outline

- `data`
  - contains the data set used for training and testing labelled with the raw runtimes, the costs when using the cost function by Healy et. al or the corresponding rankings
  - data consists of 597134 predicates that were mapped to 110 features originating from <https://www3.hhu.de/stups/downloads/prob/source/ProB_public_examples.tgz>
  - .zip files should be extracted for the Python scripts to run in their current configuration
- 'param_grids'
  - contains hyperparameters for scikit-learn used for the grid search in the form of .json files

- `results`
  - contains the output of the Python scripts **without** the binary machine learning model data (.joblib files)

- `calc_runtime_ranks_and_distr.py`
  - script that produces the [RankingDistributions.txt](results/RankingDistributions.txt)
  - accesses the files in the [data](data) directory
- `start_learners.py`
  - main script
  - performs cross validation gridsearches and evaluates the best predictors
  - contains major configuration regarding
    - algorithms to run
    - cross validation parameters
    - performance metrics
  - makes use of the Python modules in the [learning](learning) package
  - the way it works:
    - algorithm are optimized with gridsearches, cross validation performances are documented and the resulting estimators are saved as .joblib files unless an algorithm with the same name was already trained (.joblib data etc. is already present)
    - in case of the latter, a certain predictor is loaded from a file
    - all cross validation winners are then evaluated on a holdout test set, predictions are saved as .json files, confusion matrices are saved as .png files and the performances of each algorithm are documented in an evaluation file.

  - accesses files in the [data](data) and [param_grids](data) directories
  - **important:** output with the current configuration has a size of around 50GB
- `start_learners_miniconda_script.sh`
  - script for running the program on the HHU cluster
