import json
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

"""
This module manages almost any form of file access during the course of training and evaluating the machine learning
models. It is responsible for loading data from files and saving data to them.
"""


def open_file(file_dir, file_name, mode="w"):
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    return open(file_dir+os.path.sep+file_name, mode=mode)


def read_param_grids_from_files(param_grids_path, algorithm_names, param_grids):
    for algorithm_name in algorithm_names:
        algo_param_grids_path = param_grids_path + os.path.sep + "{}_param_grid.json".format(algorithm_name)
        file = open(algo_param_grids_path, "r")
        param_grids[algorithm_name] = json.load(file)
        file.close()


def get_data_from_csv(csv_src_file_path, output_dimension, random_state, train_size):
    df = pd.read_csv(csv_src_file_path)

    X = df[df.keys()[:110]]  # use column 0-119 as features
    if output_dimension == 1:
        y = df[df.keys()[110]]
    else:
        y = df[df.keys()[110:110+output_dimension]]

    X_cv, X_holdout, y_cv, y_holdout = train_test_split(X, y, random_state=random_state, train_size=train_size)

    return X_cv, X_holdout, y_cv, y_holdout


def save_all_results(cv_results, target_path, algorithm_name):
    # Save dataframes
    df = pd.DataFrame(cv_results)[['mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time', 'params']]
    df = df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)
    dataframe_file = open_file(target_path, algorithm_name + "_dataframe.txt", mode="w")
    dataframe_file.write(df.to_string())
    dataframe_file.close()
    df.to_csv(target_path + os.path.sep + algorithm_name + "_dataframe.csv")


def save_best_results(best_params, best_score, target_path, algorithm_name):
    best_results_file = open_file(target_path,
                                  "best_results_for_each_algorithm.txt", mode="a")
    print(algorithm_name, file=best_results_file)
    print("Best score: %f" % best_score, file=best_results_file)
    print(best_params, file=best_results_file)  # shows parameters grid has tested for
    print("\n", file=best_results_file)

    best_results_file.close()


def save_best_estim_and_attrib(best_estimator, best_estim_attrib, target_path, name):
    estimator_path = os.path.join(target_path, "best_" + name + "(train-set).joblib")
    joblib.dump(best_estimator, estimator_path)

    best_attributes_file = open_file(target_path, "best_" + name + "_attributes(train-set).json", mode="w")
    json.dump(best_estim_attrib, best_attributes_file)
    best_attributes_file.close()


def get_best_estim_and_attrib_tuple(algorithm_name, src_path):
    estim_file_dir = os.path.join(src_path, "best_" + algorithm_name + "(train-set).joblib")

    estimator = joblib.load(estim_file_dir)

    attr_dict_file = open_file(src_path, "best_" + algorithm_name + "_attributes(train-set).json", mode="r")
    attr_dict = json.load(attr_dict_file)
    return estimator, attr_dict


def get_best_estim_and_attrib_tuples(algorithm_names, src_path):
    tuple_list = []
    for algorithm_name in algorithm_names:
        algorithm_path = os.path.join(src_path, algorithm_name)
        estim_attrib_tuple = get_best_estim_and_attrib_tuple(algorithm_name, algorithm_path)
        tuple_list.append(estim_attrib_tuple)
    return tuple_list
