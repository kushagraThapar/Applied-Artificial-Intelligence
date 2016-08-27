This file contains the description of the Project and the necessary steps to run this project.

1. Basic steps to run this program is to have python installed on your machine.
2. This program runs on python installed by Anaconda so that you don't have to install any additional libraries.
3. I have used sklearn libraries, which comes by default when you install python on your system from Anaconda.
4. The program can be run from command line terminal.
5. Before running the program, you need to set PYTHONPATH.
6. The value of command line variable PYTHONPATH can be set as the path to this project home directory, i.e. where this readme file exists.
7. After mentioning the PYTHONPATH, you can run the main_init_.py file, which is in the src/com/main/ directory.
8. The usage of this program is like this:

python main_init_.py [training_csv_file] [test_csv_file] [store_features_directory] [Integer representing following machine learning techniques]

Naive Bayes = 1
Logistic Regression = 2
Neural Networks = 3
Random Forest = 4
Decision Trees = 5
All of Them = 6