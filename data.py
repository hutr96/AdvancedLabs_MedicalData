import numpy as np
import pandas as pd
import csv
import os

from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data = pd.read_csv("wdbc.csv")
print data.shape

length = len(data)
features = data.shape[1]-2

malignant = len(data[data['diagnosis']=='M'])
benign = len(data[data['diagnosis']=='B'])
rate = (float(malignant)/(length))*100

print "There are "+ str(len(data))+" cases in this dataset"
print "There are {}".format(features)+" features in this dataset"
print "There are {}".format(malignant)+" cases diagnosed as malignant tumor"
print "There are {}".format(benign)+" cases diagnosed as benign tumor"
print "The percentage of malignant cases is: {:.4f}%".format(rate)

data.diagnosis.unique()

data.drop('id',axis=1,inplace=True)     #drop useless feature
# print data.head(1)
features = list(data.columns[1:31])
# print features
label = data.columns[0:1]
print "Features:"
print features
print "labels: "
print label

X = data[features]
Y = data[label]
# print "\nFeature values:"
# print X.head(1)
# print "\nTarget values:"
# print Y.head(1)


# df=pd.DataFrame(data)         # looking for correlation and show them as picture
# fig, ax = plt.subplots(1)
# for i in range(1):
#     x=df['perimeter_mean']
#     y=df['area_worst']
#     ax.scatter(x,y, label=str(i))
#
# ax.set_title('Correlation of perimeter_mean and area_worst')
# fig.savefig('scatter1.png')
# fig, ax = plt.subplots(1)
# for i in range(1):
#     x=df['concavity_mean']
#     y=df['compactness_worst']
#     ax.scatter(x,y, label=str(i))
# ax.set_title('Correlation of concavity_mean and compactness_worst')
# fig.savefig('scatter2.png')


def preprocess_features(X):     # pre-process: using pandas and replace non-digital feature/label
    output = pd.DataFrame(index = X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = col_data.replace(['M', 'B'], [1, 0])
        # Collect the revised columns
        output = output.join(col_data)
    return output

X = preprocess_features(X)
Y = preprocess_features(Y)
# X=Normalizer().fit_transform(X)
# pca = PCA( n_components=5, copy=True, whiten=False)        # Dimensionality reduction  , svd_solver='full'
# X = pca.fit_transform(X)
print "shape of X :"+str(X.shape)


# split train and test set
nr_train = 400
nr_test = X.shape[0] - nr_train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nr_test, random_state=40)
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# using LDA for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=5)
X_train = lda.fit(X_train,Y_train).transform(X_train)
X_test = lda.transform(X_test)
print "shape of X_train after dimensionality reduction:"+str(X_train.shape)
print "shape of X_test after dimensionality reduction:"+str(X_test.shape)


def feature_select(X_train, X_test, y_train, n_feat='all'):
    """
    Function performs univariate feature selection using sklearn.feature_selection.SelectKBest and a score function.
    SelectKBest removes all but the "k" highest scoring features. The function will return a tuple of the reduced
    features and their respective scores and p-values.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :param n_feat:  Number of features to select
    :return: tuple (selected X_train, selected X_test, scores, p-values)
    """
    # Univariate Feature Selection - chi2 (score_function)
    score_func = SelectKBest(chi2, k=n_feat).fit(X_train, y_train) #k = # features
    select_X_train = score_func.transform(X_train)
    select_X_test = score_func.transform(X_test)

    # Score Function Attributes
    scores =  score_func.scores_
    pvalues = score_func.pvalues_
    return (select_X_train, select_X_test, scores, pvalues)

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

    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    train_classifier(clf, X_train, Y_train)

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
clf_I = MLPClassifier(activation='tanh',alpha=0.01)

clf_H = QuadraticDiscriminantAnalysis()


# feature selection     n_feat: Kbest
# select_X_train, select_X_test, score, pvalues = feature_select(X_train, X_test, Y_train)
# feat_names = list(data.columns)[:-1]
#
# print "Feature Selection"
# fselect_score = pd.concat([pd.Series(feat_names, name='feat'), pd.Series(score, name='score'),
#                                pd.Series(pvalues, name='pvalue')],axis=1)
# print fselect_score.sort_values('score', ascending=False), '\n'
#
# print "shape of selected Features"
# print select_X_train.shape


for clf in [ clf_I]:    # clf_A, clf_B, clf_C, clf_D, clf_E,
    for size in [400]:
        train_predict(clf, X_train[:size], Y_train[:size], X_test, Y_test)  #select_
        print ' '
        # if clf == clf_I:
        #     plot_roc_pp(clf, X_test, Y_test, 31, name= "{}".format(clf.__class__.__name__))




