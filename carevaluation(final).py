#SATYA SHAH
#UTA ID- 1002161494
#CSE 6332 MACHINE LEARNING ASSIGNMENT 1-KNN
#Car Evaluation

import random
from ucimlrepo import fetch_ucirepo 



# Fetching Dataset
car_evaluation = fetch_ucirepo(id=19) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 


# Label Encoding
buying_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
X_buying_encoded = [buying_mapping[val] for val in X['buying']]

maint_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
X_maint_encoded = [maint_mapping[val] for val in X['maint']]

door_mapping = {'2': 2, '3': 3, '4': 4, '5more': 5}
X_doors_encoded = [door_mapping[val] for val in X['doors']]

persons_mapping = {'2': 2, '4': 4, 'more': 7}
X_persons_encoded = [persons_mapping[val] for val in X['persons']]

lug_boot_mapping = {'big': 3, 'med': 2, 'small': 1}
X_lug_boot_encoded = [lug_boot_mapping[val] for val in X['lug_boot']]

safety_mapping = {'high': 3, 'med': 2, 'low': 1}
X_safety_encoded = [safety_mapping[val] for val in X['safety']]

class_mapping = {'vgood': 4, 'good': 3, 'acc': 2, 'unacc': 1}
y_class_encoded = [class_mapping[val] for val in y['class']]

X_encoded = list(zip(X_buying_encoded, X_maint_encoded, X_doors_encoded, X_persons_encoded, X_lug_boot_encoded, X_safety_encoded))
y_encoded = y_class_encoded





# Function to shuffle the data to randomize it
def shuffle_data(X, y, random_state=None):
    """
    Shuffle the data to randomize it.
    
    Parameters:
        X (list): The feature vectors of the dataset.
        y (list): The labels of the dataset.
        random_state (int): Random state for reproducibility.
        
    Returns:
        list: The shuffled feature vectors.
        list: The shuffled labels.
    """
    data = list(zip(X, y))
    random.seed(random_state)
    random.shuffle(data)
    X_shuffled, y_shuffled = zip(*data)
    return X_shuffled, y_shuffled



#Function to calculate std dev
def std_deviation(X, mean_values, index):
    """
    Compute the standard deviation for a feature in the dataset.
    
    Parameters:
        X (list of lists): The dataset.
        mean_values (list): The mean values of each feature.
        index (int): The index of the feature.
        
    Returns:
        float: The standard deviation of the feature.
    """
    variance = sum([(row[index] - mean_values[index]) ** 2 for row in X]) / len(X)
    return variance ** 0.5



# Function to standardize the data
def standardize_data(X):
    """
    Standardize the features of the dataset (z-score normalization).
    
    Parameters:
        X (list of lists): The dataset to be standardized.
        
    Returns:
        list of lists: The standardized dataset.
    """
    num_features = len(X[0])
    mean_values = [sum([row[i] for row in X]) / len(X) for i in range(num_features)]
    std_values = [std_deviation(X, mean_values, i) for i in range(num_features)]
    X_standardized = [[(row[i] - mean_values[i]) / std_values[i] for i in range(num_features)] for row in X]
    return X_standardized




# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Parameters:
        p1 (list): The coordinates of point 1.
        p2 (list): The coordinates of point 2.
        
    Returns:
        float: The Euclidean distance between the two points.
    """
    distance = sum((a - b) ** 2 for a, b in zip(p1, p2))
    return distance ** 0.5



# Function to calculate Manhattan distance
def manhattan_distance(p1, p2):
    """
    Calculate Manhattan distance between two points.
    
    Parameters:
        p1 (list): The coordinates of point 1.
        p2 (list): The coordinates of point 2.
        
    Returns:
        float: The Manhattan distance between the two points.
    """
    return sum(abs(a - b) for a, b in zip(p1, p2))




# Majority voting function
def majority_vote(labels):
    """
    Determines the majority class label among a list of labels.
    
    Parameters:
        labels (list): The list of class labels.
        
    Returns:
        int: The majority class label.
    """
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    majority_label = max(label_counts, key=label_counts.get)
    return majority_label



# Function to predict labels for input data using the k-nearest neighbors algorithm
def predict(X_train, y_train, X_input, k):
    """
    Predict labels for input data using the k-nearest neighbors algorithm.
    
    Parameters:
        X_train (list of lists): The feature vectors of the training data.
        y_train (list): The labels of the training data.
        X_input (list of lists): The feature vectors of the input data to be classified.
        k (int): The number of nearest neighbors to consider.
        
    Returns:
        list: The predicted labels for the input data.
    """
    predict_labels = []
    
    for item in X_input: 
        point_dist = []
        
        for j, x in enumerate(X_train): 
            distances = manhattan_distance(x, item)
            point_dist.append((distances, y_train[j])) 
         
        sorted_points = sorted(point_dist)[:k]
        labels = [label for distance, label in sorted_points]
        lab = majority_vote(labels) 
        predict_labels.append(lab)
 
    return predict_labels



# Function to calculate accuracy given the true labels and predicted labels
def calculate_accuracy(y_true, y_pred):
    """
    Calculates accuracy given the true labels and predicted labels.
    
    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        
    Returns:
        float: Accuracy score.
    """
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy



# Function to perform k-fold cross-validation with custom KNN
def k_fold_cross_validation(X, y, k, k_neighbors, random_state=None):
    accuracies = []
    indices = list(range(len(X)))
    random.seed(random_state)
    random.shuffle(indices)
    fold_size = len(X)//k
    
    for i in range(k):
        val_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = [idx for idx in indices if idx not in val_indices]

        X_train_fold, X_val_fold = [X[idx] for idx in train_indices], [X[idx] for idx in val_indices]
        y_train_fold, y_val_fold = [y[idx] for idx in train_indices], [y[idx] for idx in val_indices]

        y_pred_fold = predict(X_train_fold, y_train_fold, X_val_fold, k_neighbors)
        accuracy = calculate_accuracy(y_val_fold, y_pred_fold)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / k
    return accuracies, mean_accuracy



# Splitting and shuffling the data
random_state = 60  # Set a fixed random state for reproducibility
X_shuffled, y_shuffled = shuffle_data(X_encoded, y_encoded, random_state=random_state)


#Standardizing X
X_shuffled=standardize_data(X_shuffled)


# Performing k-fold cross-validation with custom KNN
k_folds = 10
accuracies_custom, mean_accuracy_custom = k_fold_cross_validation(X_shuffled, y_shuffled, k_folds, k_neighbors=7,random_state=random_state)
print("Mean accuracy (Custom):", mean_accuracy_custom)



### Calculating KNN from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel

# Performing k-fold cross-validation with scikit-learn KNN
knn = KNeighborsClassifier(n_neighbors=7, metric='manhattan')
accuracies_sklearn = cross_val_score(knn, X_shuffled, y_shuffled, cv=k_folds)
mean_accuracy_sklearn = sum(accuracies_sklearn) / k_folds
print("Mean accuracy (Scikit-learn):", mean_accuracy_sklearn)

# Performing paired t-test between accuracies from both methods
t_statistic, p_value = ttest_rel(accuracies_custom, accuracies_sklearn)

print("Paired t-test results:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Interpreting the p-value
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject null hypothesis: There is a statistically significant difference.")
else:
    print("Fail to reject null hypothesis: There is no statistically significant difference.")
