import numpy as np

from learning.learners.SingleLearnerLearningManager import SingleLearnerLearningManager
from learning.metrics import rank_conversion
from learning.timing import timing_context_manager
from learning.timing.timing_context_manager import time_context


class SingleRegLearnerLearningManager(SingleLearnerLearningManager):
    """
    Managing class for an instance of Multi-Output Regression,
    which means that ranking predictions are made upon the output of a regressor
    with three dimensional output.
    """

    def __init__(self, multi_output_reg_learner):
        self.is_regression = True
        name = "Single Multi-Output Regression"
        super().__init__(name, self.is_regression)
        self.multi_output_reg_learner = multi_output_reg_learner

    def learn(self):
        self.multi_output_reg_learner.learn()

    def predict(self, prediction_times_file):
        for scoring_method_name in self.multi_output_reg_learner.best_estim_and_attr_dict:
            scoring_method_dict = self.multi_output_reg_learner.best_estim_and_attr_dict[scoring_method_name]
            for algorithm_name in scoring_method_dict:
                if algorithm_name != "best":
                    (estimator, attributes) = scoring_method_dict[algorithm_name]

                    with time_context("Prediction for: " + self.name + ", " + algorithm_name + ", "
                                      + scoring_method_name, file=prediction_times_file):
                        y_holdout_pred_numeric = estimator.predict(self.multi_output_reg_learner.X_holdout)
                        y_holdout_pred_numeric = np.asarray(y_holdout_pred_numeric)
                        y_holdout_pred_ranks = rank_conversion.convert_compl_arr_to_ranks(y_holdout_pred_numeric)

                    time_to_pred_one_rank = timing_context_manager.current_elapsed_time / y_holdout_pred_ranks.size

                    self.append_pred_to_prediction_tuples(algorithm_name, scoring_method_name,
                                                          y_holdout_pred_numeric, y_holdout_pred_ranks,
                                                          attributes, time_to_pred_one_rank)
