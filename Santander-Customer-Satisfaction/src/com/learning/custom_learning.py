import pandas as pd
import numpy as np
import src.com.variables.global_variables as global_variables
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import linear_model

custom_index_array = []
predicted_data_array = []
new_target_array = []


def machine_learning(data_frame, is_train_data):
    if is_train_data:
        train_data(data_frame)
    else:
        predict_data()
    return None


def create_features_array():
    for custom_index, feature_row in global_variables.training_data_part.iterrows():
        features = []
        for feature in global_variables.features_data:
            features.append(feature_row[feature])
        global_variables.features_array.append(features)
        global_variables.target_array.append(feature_row[global_variables.TARGET_column_name])


def create_prediction_features_array():
    for custom_index, feature_row in global_variables.test_data_part.iterrows():
        features = []
        for feature in global_variables.features_data:
            features.append(feature_row[feature])
        global_variables.features_prediction_array.append(features)
        custom_index_array.append(custom_index)


def calculate_accuracy_of_predicted_data():
    print(str(len(new_target_array)) + " | " + str(len(custom_index_array)))
    i = 0
    new_count = 0
    for custom_index in custom_index_array:
        if new_target_array[i] == global_variables.test_data_part[global_variables.TARGET_column_name][custom_index]:
            new_count += 1
        i += 1

    print(new_count)
    print(new_count / len(new_target_array) * 100)
    return None


def analyze_predicted_data():
    unsatisfied_customers = 0
    satisfied_customers = 0
    for i in range(len(new_target_array)):
        if new_target_array[i] == 1:
            unsatisfied_customers += 1
        elif new_target_array[i] == 0:
            satisfied_customers += 1

    print("Unsatisfied Customers are [" + str(unsatisfied_customers) + "]")
    print("Satisfied Customers are [" + str(satisfied_customers) + "]")
    return unsatisfied_customers


def predict_data():
    print("Predicting Data Frame with technique [" + str(global_variables.machine_learning_technique) + "] ... ")
    create_prediction_features_array()
    if global_variables.machine_learning_technique == global_variables.LearningTechnique.GAUSSIAN_NB:
        apply_gaussian_nb()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.LOGISTIC_REGRESSION:
        apply_logistic_regression()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.NEURAL_NETWORKS:
        apply_neural_networks()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.RANDOM_FOREST:
        apply_random_forest()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.DECISION_TREES:
        apply_decision_trees()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.ALL:
        apply_all_machine_learning_technique()

    # calculate_accuracy_of_predicted_data()
    return None


def train_data_using_models():
    print("Training Data Frame with technique [" + str(global_variables.machine_learning_technique) + "] ... ")
    create_features_array()
    if global_variables.machine_learning_technique == global_variables.LearningTechnique.GAUSSIAN_NB:
        train_using_gaussian_nb()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.LOGISTIC_REGRESSION:
        train_using_logistic_regression()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.NEURAL_NETWORKS:
        train_using_neural_networks()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.RANDOM_FOREST:
        train_using_random_forest()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.DECISION_TREES:
        train_using_decision_trees()
    elif global_variables.machine_learning_technique == global_variables.LearningTechnique.ALL:
        train_using_all_machine_learning_technique()
    return None


def apply_gaussian_nb():
    global new_target_array
    new_target_array = global_variables.classification_gaussian_nb.predict(global_variables.features_prediction_array)
    return None


def apply_random_forest():
    global new_target_array
    new_target_array = global_variables.classification_random_forest.predict(global_variables.features_prediction_array)
    return None


def apply_decision_trees():
    global new_target_array
    new_target_array = global_variables.classification_decision_trees.predict(
        global_variables.features_prediction_array)
    return None


def apply_neural_networks():
    return None


def apply_logistic_regression():
    global new_target_array
    new_target_array = global_variables.classification_logistic_regression.predict(
        global_variables.features_prediction_array)
    return None


def apply_all_machine_learning_technique():
    return None


def train_using_gaussian_nb():
    global_variables.classification_gaussian_nb = GaussianNB()
    global_variables.classification_gaussian_nb.fit(global_variables.features_array, global_variables.target_array)
    return None


def train_using_random_forest():
    global_variables.classification_random_forest = RandomForestClassifier()
    global_variables.classification_random_forest.fit(global_variables.features_array, global_variables.target_array)
    return None


def train_using_decision_trees():
    global_variables.classification_decision_trees = tree.DecisionTreeClassifier()
    global_variables.classification_decision_trees.fit(global_variables.features_array, global_variables.target_array)
    return None


def train_using_neural_networks():
    return None


def train_using_logistic_regression():
    global_variables.classification_logistic_regression = linear_model.LogisticRegression()
    global_variables.classification_logistic_regression.fit(global_variables.features_array,
                                                            global_variables.target_array)
    return None


def train_using_all_machine_learning_technique():
    return None


def train_data(training_data_frame):
    print("Training Data ... ")
    # split_training_data(training_data_frame)
    # check_train_test_samples()
    global_variables.training_data_part = training_data_frame
    train_data_using_models()
    # predict_data()
    return None


def split_training_data(training_data_frame):
    print("Splitting Data ... ")
    global_variables.original_training_data_frame = training_data_frame
    train_ix = np.random.choice(training_data_frame.index,
                                global_variables.split_data_percentage * len(training_data_frame.index),
                                replace=False)
    global_variables.training_data_part = training_data_frame.ix[train_ix]
    global_variables.test_data_part = training_data_frame.drop(train_ix)
    return


def check_train_test_samples():
    print("Checking training and test data samples ... ")
    print(len(global_variables.original_training_data_frame.index))
    print(len(global_variables.training_data_part.index))
    print(len(global_variables.test_data_part.index))
