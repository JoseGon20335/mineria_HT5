from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from pandas.plotting import scatter_matrix
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

# Load dataset from URL
url = 'https://jupyterlite.anaconda.cloud/b0df9a1c-3954-4c78-96e6-07ab473bea1a/files/iris/iris.csv'
response = requests.get(url)
csv_data = response.text
dataset = pd.read_csv(io.StringIO(csv_data))

# Print the shape of the data
# It has 150 instances and 5 attributes
print(dataset.shape)

# Print the first 10 rows of the data
print(dataset.head(10))

# Print the last 10 rows of the data
print(dataset.tail(10))

# Describe some basic statistics about the data
print(dataset.iloc[:, 1:].describe())

# First, create a dataset backup
dataset_bak = dataset.copy()

# Remove first column - Id
dataset.drop('Id', axis=1, inplace=True)
print(dataset.head(10))

# Change column names
dataset.columns = ['Sepal-length', 'Sepal-width',
                   'Petal-length', 'Petal-width', 'Species']
print(dataset.head(20))

# Class distribution, to see the number of rows that belong to each species
print(dataset.groupby('Species').size())

# Box and whisker plots. Univariate plots, one for each individual variable
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(
    10, 5), dpi=100, facecolor='w', edgecolor='k')
dataset.plot(kind='box', subplots=True, layout=(
    2, 2), ax=axes, sharex=False, sharey=False)
plt.show()

# Histograms. Create a histogram of each input variable to get an idea of the distribution
dataset.hist()
plt.show()

# Scatter plot matrix. See all pairs of attributtes, to detect correlations or relationships
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare algorithms
plt.boxplot(results)
plt.title('Algorithm Comparison')
plt.xticks(np.arange(len(names))+1, names)
plt.show()
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
