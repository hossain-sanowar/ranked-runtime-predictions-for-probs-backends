from abc import ABC, abstractmethod


class LearningManager(ABC):
    """
    Manages a prediction model.
    An instance of a learning manager consists of at least one Learner which can be trained and used to make
    predictions.
    """

    def __init__(self, name, is_regression):
        self.name = name
        self.is_regression = is_regression
        self.prediction_tuples = []
        pass

    @abstractmethod
    def learn(self):
        """
        Trains the Learner(s) on the training set
        """
        pass

    @abstractmethod
    def predict(self, prediction_times_file):
        """
        Forces the Learner(s) to make predictions on the data in the test set
        """
        pass

    def append_pred_to_prediction_tuples(self, algorithm_name, scoring_method_name, y_holdout_pred_numeric,
                                         y_holdout_pred_ranks, attributes, time_to_pred_one_rank):
        """
        Adds the necessary info about the Learner(s) and its/their predictions to a list
        """
        self.prediction_tuples.append((self.name, algorithm_name, scoring_method_name, y_holdout_pred_numeric,
                                       y_holdout_pred_ranks, self.is_regression, attributes, time_to_pred_one_rank))
