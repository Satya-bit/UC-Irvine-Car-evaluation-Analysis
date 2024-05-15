# UC-Irvine-Car-evaluation-Analysis usi9ng KNN from scratch
Car Evaluation Database was derived from a simple hierarchical decision model originally developed for the demonstration of DEX, M. Bohanec, V. Rajkovic: Expert system for decision making. Sistemica 1(1), pp. 145-157, 1990.).  The dataset is taken from UC Irvine Machine Learning Repository

![image](https://github.com/Satya-bit/UC-Irvine-Car-evaluation-Analysis/assets/70309925/83f9d752-8a48-418e-aa67-9187ba5488a5)

# About dataset and attributes
buying:   vhigh, high, med, low.

maint:    vhigh, high, med, low.

doors:    2, 3, 4, 5more.

persons:  2, 4, more.

lug_boot: small, med, big.

safety:   low, med, high.

The class we need to predict for car is divided into four categories ('unacc', 'acc', 'good', 'vgood')
# How the code works

->Importing Libraries and Dataset Fetching:

The script begins with importing necessary libraries.
fetch_ucirepo is used to fetch a dataset from the UCI Machine Learning Repository.

->Data Preprocessing:

The script performs label encoding for categorical features using predefined mappings.
Features and labels are encoded and stored in X_encoded and y_encoded, respectively.

->Shuffling Data:

The function shuffle_data shuffles the data to randomize it, aiding in better training.

->Feature Standardization:

The function standardize_data standardizes the features of the dataset using z-score normalization.

->Distance Calculation Functions:

euclidean_distance and manhattan_distance compute Euclidean and Manhattan distances between two points, respectively.

->Majority Voting Function:

majority_vote determines the majority class label among a list of labels.

->Prediction Function:

The predict function predicts labels for input data using the KNN algorithm.

->Accuracy Calculation Function:

calculate_accuracy calculates accuracy given the true labels and predicted labels.

->k-fold Cross-Validation Function:

k_fold_cross_validation performs k-fold cross-validation with custom KNN.

->Data Shuffling and Splitting:

The data is shuffled using a fixed random state for reproducibility.

->Standardization:

The feature data is standardized using the standardize_data function.

->Custom KNN:

k-fold cross-validation is performed using the custom KNN implementation, and the mean accuracy is printed.

->Scikit-learn KNN:

Similarly, k-fold cross-validation is performed using scikit-learn's KNN implementation, and the mean accuracy is printed.

->Paired t-test:

A paired t-test is conducted between accuracies obtained from both methods to determine if there's a statistically significant difference.

->Interpreting Results:

The p-value is compared against a significance level (alpha) to determine whether to reject the null hypothesis.

# ACCURACY
![image](https://github.com/Satya-bit/UC-Irvine-Car-evaluation-Analysis/assets/70309925/5eeecdce-04a0-4677-82a0-4df17a370694)


