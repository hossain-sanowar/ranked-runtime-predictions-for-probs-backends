import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from learning.evaluation import evaluation
from learning.file_access import file_access
from learning.learners.Learner import Learner
from learning.learners.SingleClassifLearnerLearningManager import SingleClassifLearnerLearningManager
from learning.learners.SingleRegLearnerLearningManager import SingleRegLearnerLearningManager
from learning.learners.TripleLearnerRegLearningManager import TripleLearnerRegLearningManager
from learning.metrics.balanced_accuracy import balanced_accuracy_function
from learning.metrics.dndcg import dndcg_function
from learning.timing.timing_context_manager import time_context

NUMBER_OF_FOLDS = 5


def main():
    """
    Main function containing the definition of algorithms to run, configuration as well as
    calls on the training, prediction and evaluation modules and functions.
    """
    define_pandas_options()

    learning_manager_list = []
    random_state = 3012
    train_size = 0.75
    n_jobs = -2
    cv = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=random_state)

    nrmse = "neg_root_mean_squared_error"  # scikit-learn natively only supports the negative RMSE
    dndcg = make_scorer(dndcg_function, greater_is_better=True)
    balanced_accuracy = make_scorer(balanced_accuracy_function, greater_is_better=True)

    # -------------- Triple Learner Regression --------------

    single_output_reg_algorithm_tuples = [
        ("Linear_Regression", LinearRegression()),
        ("Ridge_Regression", Ridge()),
        ("Decision_Tree_Regression", DecisionTreeRegressor()),
        ("Random_Forest_Regression100", RandomForestRegressor()),
        ("Random_Forest_Regression100_max_features_auto", RandomForestRegressor()),
        ("Random_Forest_Regression100_max_features_zeropointseven", RandomForestRegressor()),
        ("Knn_Regression_uniform", KNeighborsRegressor()),
        ("Knn_Regression_distance", KNeighborsRegressor()),
        ("Linear_SVR", LinearSVR())

    ]
    single_output_reg_scoring_method_dict = {"Nrmse": nrmse}
    single_output_reg_learner_dict = {}

    for name in ["kodkod", "prob", "z3"]:
        single_reg_csv_src_file_path = os.path.join(reg_csv_src_path, name + "-healy-all.csv")
        single_output_reg_learner_dict[name] \
            = Learner(backend_name=name, learning_type="regression",
                      algorithm_tuples=single_output_reg_algorithm_tuples,
                      csv_src_file_path=single_reg_csv_src_file_path,
                      param_grids_path=reg_param_grids_path,
                      result_path=result_path, scoring_method_dict=single_output_reg_scoring_method_dict,
                      cv=cv, random_state=random_state,
                      train_size=train_size, output_dimension=1, n_jobs=n_jobs)

    triple_learner_reg_learning_manager \
        = TripleLearnerRegLearningManager(single_output_reg_learner_dict["kodkod"],
                                          single_output_reg_learner_dict["prob"],
                                          single_output_reg_learner_dict["z3"])

    learning_manager_list.append(triple_learner_reg_learning_manager)
    # -------------- Triple Learner Regression --------------

    # -------------- Single Learner Regression --------------
    multi_output_reg_algorithm_tuples = [
        ("Linear_Regression", LinearRegression()),
        ("Ridge_Regression", Ridge()),
        ("Decision_Tree_Regression", DecisionTreeRegressor()),
        ("Random_Forest_Regression100", RandomForestRegressor()),
        ("Random_Forest_Regression100_max_features_auto", RandomForestRegressor()),
        ("Random_Forest_Regression100_max_features_zeropointseven", RandomForestRegressor()),
        ("Knn_Regression_uniform", KNeighborsRegressor()),
        ("Knn_Regression_distance", KNeighborsRegressor()),
        ("Multioutput_Linear_SVR", MultiOutputRegressor(LinearSVR()))
    ]

    multi_output_reg_scoring_method_dict = {"Nrmse": nrmse, "Dndcg": dndcg, "Balanced_Accuracy": balanced_accuracy}

    combined_reg_csv_src_file_path = os.path.join(reg_csv_src_path, "combined-healy-all.csv")
    combined_reg_learner \
        = Learner(backend_name="combined", learning_type="regression",
                  algorithm_tuples=multi_output_reg_algorithm_tuples,
                  csv_src_file_path=combined_reg_csv_src_file_path,
                  param_grids_path=reg_param_grids_path,
                  result_path=result_path, scoring_method_dict=multi_output_reg_scoring_method_dict,
                  cv=cv, random_state=random_state,
                  train_size=train_size, output_dimension=3, n_jobs=n_jobs)

    single_reg_learner_learning_manager = SingleRegLearnerLearningManager(combined_reg_learner)

    learning_manager_list.append(single_reg_learner_learning_manager)

    # -------------- Single Learner Regression --------------

    # -------------- Single Learner Classification --------------

    single_output_classif_algorithm_tuples = [
        ("Ridge_Classification", RidgeClassifier()),
        ("Ridge_Classification_balanced", RidgeClassifier()),
        ("Decision_Tree_Classification", DecisionTreeClassifier()),
        ("Decision_Tree_Classification_balanced", DecisionTreeClassifier()),
        ("Random_Forest_Classification100", RandomForestClassifier()),
        ("Random_Forest_Classification100_balanced", RandomForestClassifier()),
        ("Random_Forest_Classification100_max_features_auto", RandomForestClassifier()),
        ("Random_Forest_Classification100_max_features_auto_balanced", RandomForestClassifier()),
        ("Random_Forest_Classification100_max_features_zeropointseven", RandomForestClassifier()),
        ("Random_Forest_Classification100_max_features_zeropointseven_balanced", RandomForestClassifier()),
        ("Knn_Classification_uniform", KNeighborsClassifier()),
        ("Knn_Classification_distance", KNeighborsClassifier()),
        ("Linear_SVC", LinearSVC()),
        ("Linear_SVC_balanced", LinearSVC())
    ]

    single_output_classif_method_dict = {"Dndcg": dndcg, "Balanced_Accuracy": balanced_accuracy}

    single_output_classif_csv_src_file_path = os.path.join(classif_csv_src_path, "best-ranks-healy.csv")

    single_output_classif_learner \
        = Learner(backend_name="ranking_classification", learning_type="classification",
                  algorithm_tuples=single_output_classif_algorithm_tuples,
                  csv_src_file_path=single_output_classif_csv_src_file_path,
                  param_grids_path=classif_param_grids_path,
                  result_path=result_path, scoring_method_dict=single_output_classif_method_dict,
                  cv=cv, random_state=random_state,
                  train_size=train_size, output_dimension=1, n_jobs=n_jobs)

    single_classif_learning_manager = SingleClassifLearnerLearningManager(single_output_classif_learner)

    learning_manager_list.append(single_classif_learning_manager)

    # -------------- Single Learner Classification --------------

    # Training
    with time_context("All trainings (setup, ... incl)"):
        for learning_manager in learning_manager_list:
            learning_manager.learn()

    # Prediction
    prediction_times_file = file_access.open_file(result_path, "prediction_times.txt", mode="a")
    with time_context("All predictions (setup, ... incl)"):
        for learning_manager in learning_manager_list:
            learning_manager.predict(prediction_times_file)
    prediction_times_file.close()

    # Evaluation
    y_holdout_true_numeric = get_y_holdout_values(combined_reg_csv_src_file_path, 3,
                                                  random_state, train_size)
    prediction_tuple_list = []
    prediction_tuple_list += get_standard_strategies_prediction_tuple_list(random_state, train_size)

    for learning_manager in learning_manager_list:
        prediction_tuple_list += learning_manager.prediction_tuples

    with time_context("All Evaluations"):
        evaluation.evaluate_predictions(result_path, y_holdout_true_numeric, prediction_tuple_list)


def define_pandas_options():
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.max_rows', None)  # show all rows
    pd.set_option('display.max_colwidth', None)  # show infinitely small column
    pd.options.display.width = 0  # autodetect console width(?)


def get_y_holdout_values(filepath, output_dimension, random_state, train_size):
    X_cv, X_holdout, y_cv, y_holdout \
        = file_access.get_data_from_csv(filepath, output_dimension, random_state, train_size)
    return y_holdout


def get_standard_strategies_prediction_tuple_list(random_state, train_size):
    """
    Returns a list of the standard strategies when predicting rankings:
    1. Best: Always predicts the best ranking
    2. Worst: Always predicts the worst ranking
    3. Random: Always predicts a random ranking
    """
    prediction_tuple_list = []
    for strategy_name in ["best", "worst", "random"]:
        name = "Standard strategy"
        scoring_method_name = "None"
        strat_path = os.path.join(classif_csv_src_path, strategy_name + "-ranks-healy.csv")
        y_holdout_standard_strat_numeric = None
        y_holdout_standard_strat_ranks \
            = get_y_holdout_values(strat_path, 1, random_state, train_size)
        is_regression = False
        time_to_pred_one_rank = "/"
        attributes = {"score": "/", "mean_fit_time": "/", "params": "/"}

        prediction_tuple_list.append((name, strategy_name, scoring_method_name, y_holdout_standard_strat_numeric,
                                      y_holdout_standard_strat_ranks, is_regression,
                                      attributes, time_to_pred_one_rank))
    return prediction_tuple_list


if __name__ == "__main__":
    # standard directories
    dirname = os.path.dirname
    reg_csv_src_path = os.path.join(dirname(__file__), "data", "regression", "all")
    classif_csv_src_path = os.path.join(dirname(__file__), "data", "classification", "ranks")
    reg_param_grids_path = os.path.join(dirname(__file__), "param_grids", "regression")
    classif_param_grids_path = os.path.join(dirname(__file__), "param_grids", "classification")
    result_path = os.path.join(dirname(__file__), "results", "learners", "local", "main")

    # optional alternative directories
    if len(sys.argv) == 6:
        reg_csv_src_path = sys.argv[1]
        classif_csv_src_path = sys.argv[2]
        reg_param_grids_path = sys.argv[3]
        classif_param_grids_path = sys.argv[4]
        result_path = sys.argv[5]

    with time_context("Complete Program"):
        main()
