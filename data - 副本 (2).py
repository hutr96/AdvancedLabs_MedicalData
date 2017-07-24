import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
import os
#%matplotlib inline
# import matplotlib.pyplot as plt #for plotting the graphs
# from sklearn.linear_model import LogisticRegression #for logistic regression
# from sklearn.pipeline import Pipeline #to assemble steps for cross validation
# from sklearn.preprocessing import PolynomialFeatures #for all the polynomial features
# from sklearn import svm #for Support Vector Machines
# from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier #for decision tree classifier
# from sklearn.naive_bayes import GaussianNB  #for naive bayes classifier
# from scipy import stats #for statistical info
# from sklearn.model_selection import train_test_split # to split the data in train and test
# from sklearn.model_selection import KFold # for cross validation
# from sklearn.grid_search import GridSearchCV  # for tuning parameters
# from sklearn.neighbors import KNeighborsClassifier  #for k-neighbor classifier
# from sklearn import metrics  # for checking the accuracy
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, libsvm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn import model_selection
from sklearn import grid_search
import matplotlib.pyplot as plt


data = pd.read_csv("wdbc.csv")

print data.shape

#Description of the dataset
#how many cases are included in the dataset
length = len(data)
#how many features are in the dataset
features = data.shape[1]-2

# Number of malignant cases
malignant = len(data[data['diagnosis']=='M'])

#Number of benign cases
benign = len(data[data['diagnosis']=='B'])

#Rate of malignant tumors over all cases
rate = (float(malignant)/(length))*100

print "There are "+ str(len(data))+" cases in this dataset"
print "There are {}".format(features)+" features in this dataset"
print "There are {}".format(malignant)+" cases diagnosed as malignant tumor"
print "There are {}".format(benign)+" cases diagnosed as benign tumor"
print "The percentage of malignant cases is: {:.4f}%".format(rate)

data.diagnosis.unique()

#drop ID because we do not need the ID number as shown above

data.drop('id',axis=1,inplace=True)
#check that dropped
print data.head(1)

# Extract feature columns where everything but the diagnosis is included.
# I am separating all the features that are helpful in determining the diagnosis
features = list(data.columns[1:31])
print features

#Our target is predicting the diagnosis in benign or malignant, so we need
#to extract this one as the dependent variable - the variable we will predict
label = data.columns[0:1]
print label

X = data[features] #our features that we will use to predict Y
Y = data[label] #our dependent variable, the one we are trying to predict from X
# Show the feature information by printing the first row
# Show the traget information by also printing the first row
print "\nFeature values:"
print X.head(1)
print "\nTarget values:"
print Y.head(1)


df=pd.DataFrame(data)

#Research shows that any variables that are highly correlated
#should be removed from further analysis. But, PCA takes care of multicollinearity, so maybe
#I identify them which ones there are and let PCA to do its job.
#Just in case let's see how two highly correlated variables look like
#using prettyplots

fig, ax = plt.subplots(1)
for i in range(1):
    x=df['perimeter_mean']
    y=df['area_worst']
    ax.scatter(x,y, label=str(i))
#ax.legend()
ax.set_title('Correlation of perimeter_mean and area_worst')
fig.savefig('scatter1.png')
# with correlation .99 or r-square= .81
#Let's visualize another set of variables that are not correlated as highly as the first ones
#These have a correlation coefficient of .75 which means an r-squared score of approximately .49
fig, ax = plt.subplots(1)
for i in range(1):
    x=df['concavity_mean']
    y=df['compactness_worst']
    ax.scatter(x,y, label=str(i))
#ax.legend()
ax.set_title('Correlation of concavity_mean and compactness_worst')
fig.savefig('scatter2.png')


def preprocess_features(X):

    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all M/B malignant/benign values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['M', 'B'], [1, 0])

        # Collect the revised columns
        output = output.join(col_data)

    return output

X = preprocess_features(X)
Y = preprocess_features(Y)
# print "Processed feature columns ({} total features):\n{}".format(len(X.columns), list(X.columns))
# print "Target columns ({} total features):\n{}".format(len(Y.columns), list(Y.columns))


# import cross_validation to split the train and testing
from sklearn.cross_validation import train_test_split
# Set the number of training points
nr_train = 400
# Set the number of testing points
nr_test = X.shape[0] - nr_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nr_test, random_state=40)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# train
def train_classifier(clf, X_train, Y_train):
    start = time()
    clf.fit(X_train, Y_train)
    end = time()
    print "Trained model in {:.4f} seconds".format(end - start)

#predict
def predict_labels(clf, features, label):
    start = time()
    Y_predict = clf.predict(features)
    end = time()
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(label.values, Y_predict, pos_label=1)

    #train and predict
def train_predict(clf, X_train, Y_train, X_test, Y_test):
    # Train and predict using a classifer based on F1 score. 
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, Y_train)

    # Print the results of prediction for both training and testing
    #print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, Y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, Y_test))

def gridsearch(X_train, X_test, Y_train, model):
    """
    Function determines the optimal parameters of the best classifier model/estimator by performing a grid search.
    The best model will be fitted with the Training set and subsequently used to predict the classification/labels
    of the Testing set. The function returns the "best" classifier instance, classifier predictions, best parameters,
    and grid score.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :param model: str indicating classifier model
    :return: tuple of (best classifier instance, clf predictions, dict of best parameters, grid score)
    """
    # Parameter Grid - dictionary of parameters (map parameter names to values to be searched)
    if model == 'SVM': # support vector machine
        param_grid = [
            {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
            {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']},
            {'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'degree': [2, 3, 4, 5], 'kernel': ['poly']}
        ]

        # Blank clf instance
        blank_clf = SVC()

    elif model == "RF": # random forest
        param_grid = [
            {'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'criterion': ['gini', 'entropy'],
             'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'auto']}
        ]

        rfc = RandomForestClassifier(random_state=2)

        # Grid search to find "best" random forest classifier -- Hyperparameters Optimization
        clf = model_selection.GridSearchCV(rfc, param_grid)  # classifier + optimal parameters
        clf = clf.fit(X_train, Y_train)  # fitted classifier -- Training Set
        best_est = clf.best_estimator_
        clf_pred = best_est.predict(X_test)  # apply classifier on test set for label predictions
        params = clf.best_params_  # optimal parameters
        score = clf.best_score_  # best grid score
        imp = best_est.feature_importances_
        return (best_est, clf_pred, params, score, imp)


    # Grid Search - Hyperparameters Optimization
    clf = model_selection.GridSearchCV(blank_clf, param_grid)  # classifier + optimal parameters, n_jobs=-1
    clf = clf.fit(X_train, Y_train)  # fitted classifier
    best_est = clf.best_estimator_
    clf_pred = best_est.predict(X_test)

    best_params = clf.best_params_  # best parameters identified by grid search
    score = clf.best_score_  # best grid score
    return (best_est, clf_pred, best_params, score)

# svm_model, svm_pred, svm_param, svm_score = gridsearch(X_train, X_test, Y_train, model='SVM')
# print "SVM:/n Best Parameters: ", svm_param
# print "Best Grid Search Score: ", svm_score
# print "Best Estimator: ", svm_model, "\n"
#
# rf_model, rf_pred, rf_param, rf_score, imp = gridsearch(X_train, X_test, Y_train, model='RF')
# print "RandomForest:/n Best Parameters: ", rf_param
# print "Best Grid Search Score: ", rf_score
# print "Best Estimator: ", rf_model, "\n"
def plot_roc_pp (model, X_test, target, n_features, name):
    """
    Function uses matplotlib to plot the ROC curve of the classifier.

    :param model: fitted classification model
    :param X_test: Testing set features (X_test)
    :param target: labels (y_test)
    :param n_features: int indicating number of features of data set
    :param name: str indicating classifier model
    :return: Plot of ROC curve
    """
    y_true = target
    y_score = model.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score[:,1]) #calculate FPR & TPR
    auc_score = metrics.auc(fpr, tpr) #calculate area under the curve

    # Plot
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic: (%s) \n (n_features = %d)' % (name, n_features))

    # Save Plot
    # abspath = os.path.abspath(__file__)  # absolute pathway to file
    # head_path, f_name = os.path.split(abspath)
    # work_dir = os.path.split(head_path)[0]  # root working dir
    #
    # fname = '%s_auc.png' % name
    # aucfig_path = os.path.join(work_dir, 'results', fname)
    fig.savefig('auc of {}png'.format(clf.__class__.__name__))
    return


clf_A = KNeighborsClassifier()
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = SVC(C=10,kernel='linear',gamma=0.0001)
clf_D = GaussianNB()
clf_E = RandomForestClassifier(n_estimators=10,max_features= 3, criterion='gini')
#clf_G = AdaBoostClassifier()
#clf_H = QuadraticDiscriminantAnalysis()
clf_I = MLPClassifier(alpha=1)

# X_train_100 = X_train[:100]
# Y_train_100 = Y_train[:100]
#
# X_train_200 = X_train[:200]
# Y_train_200 = Y_train[:200]

# X_train_300 = X_train[:300]
# Y_train_300 = Y_train[:300]

X_train_400 = X_train[:400]
Y_train_400 = Y_train[:400]

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_I]:
    for size in [400]:
        train_predict(clf, X_train[:size], Y_train[:size], X_test, Y_test)
        print ' '
        if clf == clf_I:
            plot_roc_pp(clf, X_test, Y_test, 30, name= "{}".format(clf.__class__.__name__))





# from itertools import cycle
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.datasets import make_classification
# from sklearn import metrics
# import pandas as pd
# from ggplot import *
#
# # ROC curve for Naive Bayes
# preds = clf_D.predict_proba(X_test)[:,1]
# fpr, tpr, _ = metrics.roc_curve(Y_test, preds)
#
# df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
# ggplot(df, aes(x='fpr', y='tpr')) +\
#     geom_line() +\
#     geom_abline(linetype='dashed')+\
#     ggtitle ("ROC for Naive Bayes has an area under the curve of " + str(metrics.auc(fpr,tpr)))
# #auc = metrics.auc(fpr,tpr)
# #ggtitle ("Area under the curve is "+ str(auc))
# #ggtitle("ROC Curve w/ AUC=%s" % str(auc))

