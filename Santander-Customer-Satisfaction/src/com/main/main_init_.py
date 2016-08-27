import sys
import src.com.variables.global_variables as global_variables
import src.com.learning.custom_learning as custom_learning
import src.com.extraction.custom_extraction as custom_extraction
import src.com.utilities.custom_utilities as custom_utilities


def main():
    args = sys.argv[1:]
    # print(args)
    if not args or len(args) != 4:
        custom_utilities.print_usage_and_exit()

    training_csv_filename = args[0]
    test_csv_filename = args[1]
    store_features_directory = args[2]
    machine_learning_technique_value = args[3]
    custom_utilities.store_machine_learning_technique(machine_learning_technique_value)
    global_variables.test_csv_filename = test_csv_filename
    global_variables.training_csv_filename = training_csv_filename

    # Extracting and Cleaning the Data
    training_data_frame = custom_extraction.process_features(training_csv_filename, store_features_directory)
    # print(training_data_frame.head())

    # Train Data Models
    custom_learning.train_data(training_data_frame)

    # Now use trained models to predict the test data

    # Extracting and Cleaning the test data
    test_data_frame = custom_extraction.process_features(test_csv_filename)
    # print(test_data_frame.head())
    # Predict the test data
    global_variables.test_data_part = test_data_frame
    custom_learning.predict_data()
    custom_learning.analyze_predicted_data()
    return


if __name__ == '__main__':
    main()
