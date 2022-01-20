from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris

"""
    This is my code but following this excellent tutorial here to refresh basics!
    https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
"""

"""--------------------[LOADING DATA]--------------------------"""
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# read_csv returns a DataFrame
dataset = read_csv(url, names=names)

# print("SHAPE:", dataset.shape)
# # See data for urself
# print(dataset.head(20))

# #All the attributes have the same scale (centimeters)
# print(dataset.describe())
# # This is nice, lets u see distribution 
# print(dataset.groupby('class').size())

 # We have a basic idea of what it looks like from ^ but let's plot it! 
"""
    Univariate plots to better understand each attribute.
    Multivariate plots to better understand the relationships between attributes.
"""
"""--------------------[VISUALIZING DATA]--------------------------"""
# Whisker plots! (Univariate plots or plot for every individual variable)
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# Histograms (Get an idea of the Distribution)
# dataset.hist()
# pyplot.show()

# Scattered matrix (Easily recognize trends!)
# scatter_matrix(dataset)
# pyplot.show()

"""--------------------[TREATING DATA]--------------------------"""
"""
    Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data.
"""

# [1] From our data, crate validation vs training set 
array = dataset.values
print("ARRAY:", array)

# Here we are going to clean the set from labels (col 5), 
X = array[:, 0:4] # all rows, columns 0-4
# and make a matching sized array for jsut the labels 
y = array[:, 4] # all rows, column 4 only 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# print("XX:", y)
# print(len(y))

# [2] use stratified 10-fold cross validation to estimate model accuracy.
"""Stratified means that each fold or split of the dataset will aim to have
 the same distribution of example by class as exist in the whole training dataset."""

# We don't know which algorithms will work best (but can get an idea by visualization step ^)
"""
    Lets try: 

    Logistic Regression (LR)
    Linear Discriminant Analysis (LDA)
    K-Nearest Neighbors (KNN).
    Classification and Regression Trees (CART).
    Gaussian Naive Bayes (NB).
    Support Vector Machines (SVM).
"""

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#[3]Eval models!
results = [] 
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()