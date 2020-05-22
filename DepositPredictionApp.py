import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus as pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# READING DATA
BankDF = pd.read_csv('bank_customer.csv', delimiter=',')
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART A -- Combining Values
print("(PART A)\n")


def combineValues(df, searchingAttribute, searchingValues, newValue):
    df.loc[df[searchingAttribute].isin(searchingValues), searchingAttribute] = newValue
    return df


# Combining JOB's
combineValues(BankDF, 'job', ['management', 'admin.'], 'white-collar')
combineValues(BankDF, 'job', ['services', 'housemaid'], 'pink-collar')
combineValues(BankDF, 'job', ['retired', 'student', 'unemployed', 'unknown'], 'other')
print("(Combined Jobs)\n", BankDF.groupby('job').agg({'job': 'count'}))

# Combining POUTCOME's
combineValues(BankDF, 'poutcome', ['retired', 'student', 'unemployed', 'unknown'], 'other')
print("\n(Combined Poutcomes)\n", BankDF.groupby('poutcome').agg({'poutcome': 'count'}))

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART B -- Converting to numerical values
print("(PART B)\n")


def labelEncoder(df):
    attributeArray = list(df.select_dtypes(include=['object']))
    encoder = LabelEncoder()
    for attribute in attributeArray:
        df[attribute] = encoder.fit_transform(df[attribute])

    return df


labelEncoder(BankDF)

print("(Converted Categorical values to Numerical)\n", BankDF.head())

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART C -- Splitting into train/test set
print("(PART C)\n")

x = BankDF.drop('deposit', axis=1)
y = BankDF.deposit

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
# train_features, test_features, train_target, test_target
print("(Shapes of Train and Test Attributes)")
print(X_train.shape)
print(X_test.shape)

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART D -- Creating subdata with selected attributes
print("(PART D)\n")

data1_columns = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'duration', 'poutcome']
data_1_train_attribute = X_train[data1_columns]
data_1_test_attribute = X_test[data1_columns]

data2_columns = ['job', 'marital', 'education', 'housing']
data_2_train_attribute = X_train[data2_columns]
data_2_test_attribute = X_test[data2_columns]

print("Train and Test sets of data_1 and data_2 are created...")

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART E F -- Training decision tree using entropy and gini index
print("(PART E\F)\n")


def train(train_feature, test_feature, method, given_depth=None):
    tree = DecisionTreeClassifier(criterion=method, max_depth=given_depth, random_state=123).fit(train_feature,
                                                                                                 y_train)
    prediction = tree.predict(test_feature)
    return metrics.accuracy_score(y_test, prediction), tree


print("Accuracies for (ENTROPY) with default depth limits.\n")
print("The accuracy for data_1 using (Entropy) is ",
      train(data_1_train_attribute, data_1_test_attribute, "entropy")[0])
print("The accuracy for data_2 using (Entropy) is ",
      train(data_2_train_attribute, data_2_test_attribute, "entropy")[0])
print("\n----------------------------------------------------------\n")

print("Accuracies for (GINI) with default depth limits.\n")
print("The accuracy for data_1 using (Gini) is ",
      train(data_1_train_attribute, data_1_test_attribute, "gini")[0])
print("The accuracy for data_2 using (Gini) is ",
      train(data_2_train_attribute, data_2_test_attribute, "gini")[0])

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART G -- Pruning the data by changing 'depth limits'
print("(PART G)\n")


def find_optimum_depth(train_feature, test_feature, method):
    max_accuracy = 0
    tree = None
    accuracies = []
    for depth in range(1, 16):

        accuracy, current_tree = train(train_feature, test_feature, method, depth)
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            tree = current_tree

    return max_accuracy, accuracies, tree


def plot_accuracies(accuracies, title):
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plt.plot(depths, accuracies)
    plt.suptitle(title)
    plt.ylabel("Accuracies")
    plt.xlabel("Depth Limits")
    plt.show()
    return


print("Accuracies for (ENTROPY) with optimum depth limits.\n")
data1_entropy, data1_entropy_accuracies, data1_entropy_tree, = find_optimum_depth(data_1_train_attribute,
                                                                                  data_1_test_attribute, "entropy")
print("The accuracy for data_1 using (Entropy) with optimum depth  is ", data1_entropy)
plot_accuracies(data1_entropy_accuracies, "Data1 Entropy Accuracies")
# ---------------------------------------------------------------------
data2_entropy, data2_entropy_accuracies, data2_entropy_tree = find_optimum_depth(data_2_train_attribute,
                                                                                 data_2_test_attribute, "entropy")
print("The accuracy for data_2 using (Entropy) with optimum depth  is ", data2_entropy)
plot_accuracies(data2_entropy_accuracies, "Data2 Entropy Accuracies")

print("\n----------------------------------------------------------\n")

print("Accuracies for (GINI) with optimum depth limits.\n")
data1_gini, data1_gini_accuracies, data1_gini_tree = find_optimum_depth(data_1_train_attribute, data_1_test_attribute,
                                                                        "gini")
print("The accuracy for data_1 using (Gini) with optimum depth  is ", data1_gini)
plot_accuracies(data1_gini_accuracies, "Data1 Gini Accuracies")
# ---------------------------------------------------------------------
data2_gini, data2_gini_accuracies, data2_gini_tree = find_optimum_depth(data_2_train_attribute, data_2_test_attribute,
                                                                        "gini")
print("The accuracy for data_2 using (Gini) with optimum depth  is ", data2_gini)
plot_accuracies(data2_gini_accuracies, "Data2 Gini Accuracies")

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART H -- Finding the lower and upper limits of "p" by calculating the confidence interval
print("(PART H)\n")


# In this part I will be using the given method in our lesson:
# p = (2*N*acc + Z**2 +- np.sqrt(Z**2 + 4*N*acc - 4*N*acc*acc)) / (2*(N+Z**2))

def confidence_interval_for_acc(acc, up):
    N = len(y_test)
    Z = 1.96
    p = 0.0
    if up:
        p = (2 * N * acc + Z ** 2 + np.sqrt(Z ** 2 + 4 * N * acc - 4 * N * acc * acc)) / (2 * (N + Z ** 2))

    else:
        p = (2 * N * acc + Z ** 2 - np.sqrt(Z ** 2 + 4 * N * acc - 4 * N * acc * acc)) / (2 * (N + Z ** 2))
    return p * 100


# Data1 - Entropy - Upper/Lower
print("(Upper) (p) value for (Data1) using (Entropy) is : ", confidence_interval_for_acc(data1_entropy, True), "%")
print("(Lower) (p) value for (Data1) using (Entropy) is : ", confidence_interval_for_acc(data1_entropy, False), "%")
print("-------")
# Data2 - Entropy - Upper/Lower
print("(Upper) (p) value for (Data2) using (Entropy) is : ", confidence_interval_for_acc(data2_entropy, True), "%")
print("(Lower) (p) value for (Data2) using (Entropy) is : ", confidence_interval_for_acc(data2_entropy, False), "%")
print("***************")
# Data1 - Gini - Upper/Lower
print("(Upper) (p) value for (Data1) using (Gini) is : ", confidence_interval_for_acc(data1_gini, True), "%")
print("(Lower) (p) value for (Data1) using (Gini) is : ", confidence_interval_for_acc(data1_gini, False), "%")
print("-------")
# Data2 - Gini - Upper/Lower
print("(Upper) (p) value for (Data2) using (Gini) is : ", confidence_interval_for_acc(data2_gini, True), "%")
print("(Lower) (p) value for (Data2) using (Gini) is : ", confidence_interval_for_acc(data2_gini, False), "%")
print("***************")

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART I -- Displaying the decision trees obtained in steps e and f.
print("(PART I)\n")


def display_decision_tree(given_tree, columns, png_name):
    dot_data = StringIO()
    export_graphviz(data1_entropy_tree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=data1_columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(png_name)
    Image(graph.create_png())
    print("Decision tree is created as ", png_name)
    return


print("Decision Tree's are being created...")

display_decision_tree(data1_entropy_tree, data1_columns, "data1_entropy_tree.png")

display_decision_tree(data2_entropy_tree, data2_columns, "data2_entropy_tree.png")

display_decision_tree(data1_gini_tree, data1_columns, "data1_gini_tree.png")

display_decision_tree(data2_gini_tree, data2_columns, "data2_gini_tree.png")

print("All Tree's are created.")

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------
