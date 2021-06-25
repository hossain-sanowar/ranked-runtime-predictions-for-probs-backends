from sklearn import preprocessing

from learning.learners.SingleLearnerLearningManager import SingleLearnerLearningManager
from learning.metrics import rank_conversion
from learning.timing import timing_context_manager
from learning.timing.timing_context_manager import time_context


class SingleClassifLearnerLearningManager(SingleLearnerLearningManager):
    """
    Managing class for an instance of Single-Output Classification,
    representing the approach of ranking classification.
    """

    def __init__(self, classif_learner):
        is_regression = False
        name = "Single Output Classification"
        super().__init__(name, is_regression)
        self.classif_learner = classif_learner

        self.le = preprocessing.LabelEncoder()
        self.le.fit(rank_conversion.RANK_LIST)

        self.y_cv_ranks = classif_learner.y_cv
        self.y_cv_numeric_labels = self.le.transform(self.y_cv_ranks)

    def learn(self):
        self.classif_learner.y_cv = self.y_cv_numeric_labels
        self.classif_learner.learn()
        self.classif_learner.y_cv = self.y_cv_ranks

    def predict(self, prediction_times_file):
        for scoring_method_name in self.classif_learner.best_estim_and_attr_dict:
            scoring_method_dict = self.classif_learner.best_estim_and_attr_dict[scoring_method_name]
            for algorithm_name in scoring_method_dict:
                if algorithm_name != "best":
                    (estimator, attributes) = scoring_method_dict[algorithm_name]

                    with time_context("Prediction for: " + self.name+", " + algorithm_name+", " + scoring_method_name,
                                      file=prediction_times_file):
                        y_holdout_pred_numeric_labels = estimator.predict(self.classif_learner.X_holdout)
                    y_holdout_pred_ranks = self.le.inverse_transform(y_holdout_pred_numeric_labels)

                    y_holdout_pred_numeric = None
                    time_to_pred_one_rank = timing_context_manager.current_elapsed_time / y_holdout_pred_ranks.size

                    self.append_pred_to_prediction_tuples(algorithm_name, scoring_method_name,
                                                          y_holdout_pred_numeric, y_holdout_pred_ranks,
                                                          attributes, time_to_pred_one_rank)
