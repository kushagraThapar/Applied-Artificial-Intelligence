import sys
import pandas as pd
import src.com.variables.global_variables as global_variables
import src.com.learning.custom_learning as custom_learning


def read_csv_file(filename):
    try:
        training_data = pd.read_csv(filename)
        return training_data
    except OSError:
        print("[" + filename + "]file not found. The program will exit now.")
        sys.exit(1)


def write_file(filename, text):
    f = open(filename, "w")
    f.write(text)
    f.close()
    return


def read_file(filename):
    try:
        f = open(filename, "rU")
        text = f.read()
        f.close()
        return text
    except FileNotFoundError:
        print("[" + filename + "]file not found. The program will exit now.")
        sys.exit(1)


def print_usage_and_exit():
    print('\n\nusage: main_init_.py [training_csv_file] [test_csv_file] [store_features_directory] '
          '[Integer representing following machine learning techniques]\n')
    print("Naive Bayes = 1\nLogistic Regression = 2\n"
          "Neural Networks = 3\nRandom Forest = 4\n"
          "Decision Trees = 5\nAll of Them = 6")
    sys.exit(1)


def store_machine_learning_technique(machine_learning_technique_value):
    try:
        machine_learning_technique_value = int(machine_learning_technique_value)
    except ValueError:
        print("\nPlease enter Machine Learning Technique as an integer ranging from 1 to 6")
        print_usage_and_exit()

    if machine_learning_technique_value < 1 or machine_learning_technique_value > 6:
        print_usage_and_exit()

    global_variables.machine_learning_technique = global_variables.LearningTechnique(machine_learning_technique_value)
    dump_machine_learning_technique()


def dump_machine_learning_technique():
    print(global_variables.machine_learning_technique)


# def set_test_data_flag(str_value):
#     value = str_value.lower()
#     value = value.capitalize()
#     if value == "True":
#         global is_test_data
#         is_test_data = True
#     elif value == "False":
#         global is_test_data
#         is_test_data = False
#     else:
#         print('usage: main_init_.py [training_csv_file/test_csv_file] [store_features_directory] '
#               '<optional: Boolean value [True / False] for "Is Test Data">')
#         sys.exit(1)
