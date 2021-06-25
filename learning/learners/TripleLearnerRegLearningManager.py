import time
from learning.learners.LearningManager import LearningManager
from learning.metrics import rank_conversion
from learning.timing import timing_context_manager
from learning.timing.timing_context_manager import time_context
import numpy as np


class TripleLearnerRegLearningManager(LearningManager):
    """
    Managing class for an instance of Separate Single-Output Regression,
    which means that ranking predictions are made upon the combined output of
    three regressors.
    """
    def __init__(self, kodkod_learner, prob_learner, z3_learner):
        is_regression = True
        name = "Separate Single-Output Regression"
        super().__init__(name, is_regression)

        self.kodkod_learner = kodkod_learner
        self.prob_learner = prob_learner
        self.z3_learner = z3_learner

    def learn(self):
        self.kodkod_learner.learn()
        self.prob_learner.learn()
        self.z3_learner.learn()

    def predict(self, prediction_times_file):
        for scoring_method_name in self.kodkod_learner.best_estim_and_attr_dict:
            algorithm_name = "best"

            with time_context("Prediction for: " + self.name + ", " + algorithm_name + ", " + scoring_method_name,
                              file=prediction_times_file):
                best_estim_kodkod, best_estim_attrib_kodkod \
                    = self.kodkod_learner.best_estim_and_attr_dict[scoring_method_name][algorithm_name]
                y_holdout_pred_kodkod = best_estim_kodkod.predict(self.kodkod_learner.X_holdout)

                best_estim_prob, best_estim_attrib_prob \
                    = self.prob_learner.best_estim_and_attr_dict[scoring_method_name][algorithm_name]
                y_holdout_pred_prob = best_estim_prob.predict(self.prob_learner.X_holdout)

                best_estim_z3, best_estim_attrib_z3 \
                    = self.z3_learner.best_estim_and_attr_dict[scoring_method_name][algorithm_name]
                y_holdout_pred_z3 = best_estim_z3.predict(self.z3_learner.X_holdout)

                y_holdout_pred_numeric = list(zip(y_holdout_pred_kodkod, y_holdout_pred_prob, y_holdout_pred_z3))
                y_holdout_pred_numeric = np.asarray(y_holdout_pred_numeric)
                y_holdout_pred_ranks = rank_conversion.convert_compl_arr_to_ranks(y_holdout_pred_numeric)

            score_list = [best_estim_attrib_kodkod["score"], best_estim_attrib_prob["score"],
                          best_estim_attrib_z3["score"]]

            param_list = [best_estim_attrib_kodkod["params"], best_estim_attrib_prob["params"],
                          best_estim_attrib_z3["params"]]

            mean_fit_time = (best_estim_attrib_kodkod["mean_fit_time"] + best_estim_attrib_prob["mean_fit_time"]
                             + best_estim_attrib_z3["mean_fit_time"])

            attributes = {"score": score_list, "params": param_list, "mean_fit_time": mean_fit_time}

            time_to_pred_one_rank = timing_context_manager.current_elapsed_time / y_holdout_pred_ranks.size

            self.append_pred_to_prediction_tuples(algorithm_name, scoring_method_name, y_holdout_pred_numeric,
                                                  y_holdout_pred_ranks, attributes, time_to_pred_one_rank)

