from enum import Enum


class LearningTechnique(Enum):
    GAUSSIAN_NB = 1
    LOGISTIC_REGRESSION = 2
    NEURAL_NETWORKS = 3
    RANDOM_FOREST = 4
    DECISION_TREES = 5
    ALL = 6


# is_test_data = False
training_csv_filename = ""
test_csv_filename = ""
store_features_filename = ""

features_data = list()
split_data_percentage = 0.8
training_data_part = None
test_data_part = None
original_training_data_frame = None
machine_learning_technique = LearningTechnique.ALL

ID_column_name = "ID"
TARGET_column_name = "TARGET"
features_filename = "features_names.txt"

features_array = []
target_array = []
features_prediction_array = []
classification_gaussian_nb = None
classification_random_forest = None
classification_decision_trees = None
classification_neural_networks = None
classification_logistic_regression = None
