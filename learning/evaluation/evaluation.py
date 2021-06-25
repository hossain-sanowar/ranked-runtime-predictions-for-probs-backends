import itertools
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, confusion_matrix
from learning.file_access import file_access
from learning.metrics import rank_conversion, resulting_execution_time_conversion
from learning.metrics.dndcg import get_dndcg_array
from start_learners import NUMBER_OF_FOLDS


def evaluate_predictions(result_path, y_holdout_true_numeric, prediction_tuple_list):
    """
    Writes the predictions on the data in the test set by the different Learning Managers as well as the resulting
    confusion matrices to files.
    Evaluates all cross validation winners with respect to their corresponding UDC, dnDCG, BAC and RMSE scores and
    writes the results to the evaluation.txt and evaluation.csv files
    """

    # Ground Truth calculations
    y_holdout_true_numeric = np.asarray(y_holdout_true_numeric)
    y_holdout_true_ranks = rank_conversion.convert_compl_arr_to_ranks(y_holdout_true_numeric)

    y_holdout_true_times = resulting_execution_time_conversion\
        .get_times_as_array(y_holdout_true_ranks, y_holdout_true_numeric)
    y_holdout_true_total_time = y_holdout_true_times.sum()

    dump_predictions(result_path, "", "", "",
                     "y_holdout_true_numeric", y_holdout_true_numeric)
    dump_predictions(result_path, "", "", "",
                     "y_holdout_true_ranks", y_holdout_true_ranks)

    df_rows = []

    # Machine Learning model evaluations
    for (name, algorithm_name, scoring_method_name, y_holdout_pred_numeric, y_holdout_pred_ranks, is_regression,
         attributes, time_to_pred_one_rank) \
            in prediction_tuple_list:

        dump_predictions(result_path, name, algorithm_name, scoring_method_name,
                         "y_holdout_pred_ranks", y_holdout_pred_ranks)
        save_confusion_matrices(result_path, name, algorithm_name, scoring_method_name, y_holdout_true_ranks,
                                y_holdout_pred_ranks)

        rmse = "/"
        if is_regression:
            learning_type = "regression"
            rmse = mean_squared_error(y_holdout_true_numeric, y_holdout_pred_numeric, squared=False)

            dump_predictions(result_path, name, algorithm_name, scoring_method_name,
                             "y_holdout_pred_numeric", y_holdout_pred_numeric)
        else:
            learning_type = "classification"

        y_holdout_pred_times = resulting_execution_time_conversion\
            .get_times_as_array(y_holdout_pred_ranks, y_holdout_true_numeric)
        y_holdout_pred_total_time = y_holdout_pred_times.sum()
        udc = y_holdout_pred_total_time / y_holdout_true_total_time

        dndcg_arr = get_dndcg_array(y_holdout_true_ranks, y_holdout_pred_ranks)
        dndcg_mean = dndcg_arr.mean()
        dndcg_std = dndcg_arr.std()

        bal_acc = balanced_accuracy_score(y_holdout_true_ranks, y_holdout_pred_ranks)

        df_rows.append((name, learning_type, algorithm_name, scoring_method_name,
                        attributes["score"],
                        time_to_pred_one_rank,
                        udc, dndcg_mean, dndcg_std, bal_acc,
                        rmse,
                        NUMBER_OF_FOLDS * attributes["mean_fit_time"], attributes["params"]))

    # save as dataframe
    df = pd.DataFrame(df_rows, columns=["Name", "Learning type", "Algorithm name", "Scoring(train-set)",
                                        "Mean score(train-set)",
                                        "Mean pred. time(one rank)(sec)",
                                        "UDC", "Dndcg-mean", "Dndcg-std", "Balanced Accuracy",
                                        "RMSE",
                                        "Fit time(train-set)(sec)", "Hyperparameters"])

    df.to_csv(os.path.join(result_path, "evaluation.csv"))
    evaluation_text_file = file_access.open_file(result_path, "evaluation.txt", mode="w")
    evaluation_text_file.write(df.to_string())
    evaluation_text_file.close()


def dump_predictions(result_path, manager_name, algorithm_name, scoring_method_name, list_name, y_pred_arr):
    """
    Saves an array of predictions as a .json file
    """
    path = os.path.join(result_path, "pred_and_conf_mat", manager_name, algorithm_name,
                        scoring_method_name)
    list_file = file_access.open_file(path, list_name + ".json", mode="w")
    json.dump(y_pred_arr.tolist(), list_file)
    list_file.close()


# The following two functions are heavily inspired by:
# https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
def save_confusion_matrices(result_path, manager_name, algorithm_name, scoring_method_name, y_true_ranks, y_pred_ranks):
    path = os.path.join(result_path, "predictions_and_confusion_matrices", manager_name, algorithm_name,
                        scoring_method_name)
    """
    Saves the confusion matrices as .png files
    """

    labels = ["kpz", "kzp", "pkz", "pzk", "zkp", "zpk"]
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true_ranks, y_pred_ranks)

    # Plot non-normalized confusion matrix

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix, without normalization')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix.png"))
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix_normalized.png"))
    plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, precision=2):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    cm = np.round(cm, precision)
    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
