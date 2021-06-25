import os
from sklearn.model_selection import GridSearchCV
from learning.file_access import file_access
from learning.timing.timing_context_manager import time_context


class Learner:
    """
    Represents the instance of a machine learning algorithm that can be trained.
    """

    def __init__(self, backend_name, learning_type, algorithm_tuples, csv_src_file_path,
                 param_grids_path, result_path, scoring_method_dict, cv, random_state=3012, train_size=0.75,
                 output_dimension=1, n_jobs=1):
        # main fields
        self.backend_name = backend_name
        self.learning_type = learning_type
        self.algorithm_tuples = algorithm_tuples
        self.algorithm_names, estim_objects = zip(*self.algorithm_tuples)
        # dictionary storing the cross validation winners and their essential attributes
        # (score, parameters and the average time needed for training)
        self.best_estim_and_attr_dict = {}

        # Paths
        self.csv_src_file_path = csv_src_file_path
        self.param_grids_path = param_grids_path
        self.backend_result_path = os.path.join(result_path, self.learning_type, self.backend_name)
        self.scoring_method_result_path = self.backend_result_path

        # Sklearn
        self.cv = cv
        self.random_state = random_state
        self.train_size = train_size
        self.output_dimension = output_dimension
        self.scoring_method_dict = scoring_method_dict
        self.n_jobs = n_jobs

        # retrieve in and output data
        self.X_cv, self.X_holdout, self.y_cv, self.y_holdout \
            = file_access.get_data_from_csv(self.csv_src_file_path, self.output_dimension,
                                            self.random_state, self.train_size)

    def learn(self):
        # get param grids from files
        param_grids = {}
        file_access.read_param_grids_from_files(self.param_grids_path, self.algorithm_names, param_grids)

        for scoring_method_name in self.scoring_method_dict:
            scoring_method = self.scoring_method_dict[scoring_method_name]
            self.scoring_method_result_path = os.path.join(self.backend_result_path, scoring_method_name)
            self.best_estim_and_attr_dict[scoring_method_name] = {}

            if not self.best_estim_and_attr_already_exist():
                name_grid_tuples = []
                # initialize grids
                for (algorithm_name, algorithm_object) in self.algorithm_tuples.copy():
                    grid = GridSearchCV(algorithm_object, param_grids[algorithm_name],
                                        cv=self.cv, n_jobs=self.n_jobs, scoring=scoring_method)
                    name_grid_tuples.append((algorithm_name, grid))

                timer_values_file = file_access.open_file(self.scoring_method_result_path, "times.txt", mode="a")

                # grid search for all algorithms
                for (algorithm_name, grid) in name_grid_tuples:
                    if not self.best_algorithm_estim_and_attr_already_exist(algorithm_name):
                        # CV-Training
                        with time_context("Training for: " + self.backend_name + ", "
                                          + scoring_method_name + ", " + algorithm_name,
                                          file=timer_values_file):
                            grid.fit(self.X_cv, self.y_cv)

                        self.save_results(grid, scoring_method_name, algorithm_name)
                    else:
                        self.load_best_algorithm_estim_and_attr_from_file(scoring_method_name, algorithm_name)
                        print(self.backend_name + ", " + scoring_method_name + ", " + algorithm_name
                              + " skipped, best estimator and its attributes are already present. "
                              + "They were loaded from a file.")

                self.determine_and_save_scoring_method_best_estim_and_attrib_tuple(scoring_method_name)

                print(self.backend_name + ", " + scoring_method_name
                      + ": Best estimator and its attributes were successfully determined.")

                timer_values_file.close()
            else:
                for algorithm_name in self.algorithm_names:
                    self.load_best_algorithm_estim_and_attr_from_file(scoring_method_name, algorithm_name)
                self.load_best_estim_and_attr_from_file(scoring_method_name)
                print(self.backend_name + ", " + scoring_method_name
                      + ": skipped, best estimator and its attributes are already present. "
                      + "They were loaded from a file.")

    def save_results(self, grid, scoring_method_name, algorithm_name):
        algorithm_result_path = os.path.join(self.scoring_method_result_path, algorithm_name)

        file_access.save_all_results(grid.cv_results_, algorithm_result_path, algorithm_name)
        file_access.save_best_results(grid.best_params_, grid.best_score_, self.scoring_method_result_path,
                                      algorithm_name)

        best_estimator_mean_fit_time = grid.cv_results_['mean_fit_time'][grid.best_index_]
        best_estim_attr = {"score": grid.best_score_, "params": grid.best_params_,
                           "mean_fit_time": best_estimator_mean_fit_time}

        self.best_estim_and_attr_dict[scoring_method_name][algorithm_name] = (grid.best_estimator_, best_estim_attr)
        file_access.save_best_estim_and_attrib(grid.best_estimator_, best_estim_attr,
                                               algorithm_result_path, algorithm_name)

    def best_algorithm_estim_and_attr_already_exist(self, algorithm_name):
        best_algorithm_estim_path = os.path.join(self.scoring_method_result_path, algorithm_name,
                                                 "best_" + algorithm_name + "(train-set).joblib")
        best_algorithm_attr_path = os.path.join(self.scoring_method_result_path, algorithm_name,
                                                "best_" + algorithm_name + "_attributes(train-set).json")
        both_exist = os.path.exists(best_algorithm_estim_path) and os.path.exists(best_algorithm_attr_path)

        return both_exist

    def best_estim_and_attr_already_exist(self):
        best_estim_path = os.path.join(self.scoring_method_result_path,
                                       "best_estimator(train-set).joblib")
        best_attr_path = os.path.join(self.scoring_method_result_path,
                                      "best_estimator_attributes(train-set).json")
        both_exist = os.path.exists(best_estim_path) and os.path.exists(best_attr_path)

        return both_exist

    @staticmethod
    def get_best_estim_and_attr_tuple_for_current_scoring_method(backend_best_estimator_and_attrib_tuples):
        best_estim = None
        best_attrib = None
        max_score = -float("inf")
        for (estim, attrib) in backend_best_estimator_and_attrib_tuples:
            score = attrib["score"]
            if score > max_score:
                best_estim, best_attrib, max_score = estim, attrib, score
        return best_estim, best_attrib

    def determine_and_save_scoring_method_best_estim_and_attrib_tuple(self, scoring_method_name):
        scoring_method_best_estim_and_attrib_tuples \
            = file_access.get_best_estim_and_attrib_tuples(self.algorithm_names, self.scoring_method_result_path)
        self.best_estim_and_attr_dict[scoring_method_name]["best"] \
            = Learner.get_best_estim_and_attr_tuple_for_current_scoring_method(
            scoring_method_best_estim_and_attrib_tuples)

        best_estimator = self.best_estim_and_attr_dict[scoring_method_name]["best"][0]
        best_estim_attrib = self.best_estim_and_attr_dict[scoring_method_name]["best"][1]
        file_access.save_best_estim_and_attrib(best_estimator, best_estim_attrib,
                                               self.scoring_method_result_path, "estimator")

    def load_best_algorithm_estim_and_attr_from_file(self, scoring_method_name, algorithm_name):
        algorithm_path = os.path.join(self.scoring_method_result_path, algorithm_name)
        self.best_estim_and_attr_dict[scoring_method_name][algorithm_name] \
            = file_access.get_best_estim_and_attrib_tuple(algorithm_name, algorithm_path)

    def load_best_estim_and_attr_from_file(self, scoring_method_name):
        self.best_estim_and_attr_dict[scoring_method_name]["best"] \
            = file_access.get_best_estim_and_attrib_tuple("estimator", self.scoring_method_result_path)