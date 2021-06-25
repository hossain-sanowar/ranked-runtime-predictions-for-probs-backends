from learning.learners.LearningManager import LearningManager


class SingleLearnerLearningManager(LearningManager):

    def __init__(self, name, is_regression):
        super().__init__(name, is_regression)

    def learn(self):
        pass

    def predict(self, prediction_times_file):
        pass
