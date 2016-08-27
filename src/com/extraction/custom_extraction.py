import src.com.variables.global_variables as global_variables
import src.com.utilities.custom_utilities as custom_utilities
import src.com.cleaning.custom_cleaning as custom_cleaning


def process_features(csv_filename, store_features_directory=None):
    print("Processing File [" + csv_filename + "] ... ")
    if store_features_directory is not None:
        print("Storing features in a file ... ")
        store_features(csv_filename, store_features_directory)
    data_frame = set_column_values(csv_filename)

    # Cleaning the Data
    custom_cleaning.clean_data(data_frame)
    return data_frame


def set_column_values(csv_filename):
    data_frame = custom_utilities.read_csv_file(csv_filename)
    features_list = []
    features_list.extend([global_variables.ID_column_name] + global_variables.features_data)
    if len(data_frame.columns) == 371:
        features_list.extend([global_variables.TARGET_column_name])

    data_frame.columns = features_list
    data_frame = data_frame.set_index(["ID"]).sort_index()
    return data_frame


def store_features(csv_filename, store_features_directory):
    train_data = custom_utilities.read_csv_file(csv_filename)
    global_variables.features_data = list(train_data.columns.values)[1:-1]
    global_variables.store_features_filename = store_features_directory + "/" + global_variables.features_filename
    features_str = ""
    for features in global_variables.features_data:
        features_str += features + "\n"
    features_str = features_str[:-1]
    custom_utilities.write_file(global_variables.store_features_filename, features_str)
    return
