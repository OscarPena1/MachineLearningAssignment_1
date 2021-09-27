import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler

#Establish a seed value
seed = 0

#Helpful Functions
######################################################
#Function to plot learning curves
#Source: Scikit Learn webpage.
#Article "Plotting Learning Curves"
#Code retrieved from 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.0, 1.01), cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    print("Train score means: " + str(train_scores_mean))
    print("Test score means: " + str(test_scores_mean))
    print("")
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fitting Times (seconds)")
    axes[1].set_title("Scalability of the Model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fitting Times (seconds)")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the Model")

    return plt

######################################################
#Function to plot validation curves
#Source: Scikit Learn webpage.
#Article "Plotting Validation Curves"
#Code modified from 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
def plot_validation_curve(classifier, train_x, train_y, par, hyper, metric, cv_choice, classifier_type, hyper_name):
    
    train_scores, test_scores = validation_curve(classifier, train_x, train_y, param_name=par, param_range=hyper, scoring=metric, n_jobs=-1, cv=cv_choice)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
#    print(train_scores)
#    print(train_scores_mean)
    
    ind=np.argmax(np.mean(test_scores, axis=1), axis=0)
    
#    print(hyper[ind])
    
    plt.title("Validation Curve with {0}".format(classifier_type))
    plt.xlabel("{name}".format(name=hyper_name))
    plt.ylabel("Score - {}".format(metric))
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(hyper, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(hyper, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(hyper, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(hyper, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")

    plt.annotate("Best Value: {0}".format(hyper[ind]),
            xy=(hyper[ind],test_scores_mean[ind]), xycoords='data',
            xytext=(20, -30), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
    plt.show()

###############################################################

#Create two random sets for classification
full_1_X, full_1_y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, random_state=seed, class_sep=.1, shuffle=True)
full_2_X, full_2_y = make_classification(n_samples=8000, n_features=10, n_informative=2, n_redundant=1, n_clusters_per_class=2, random_state=seed, flip_y=.3, class_sep=.5, shuffle=True, weights=[0.7,0.3])


#Scale the data for easier processing by algorithm
min_max_scaler =  MinMaxScaler()
full_1_X_minmax = min_max_scaler.fit_transform(full_1_X)
full_2_X_minmax = min_max_scaler.fit_transform(full_2_X)

#Show interesting features of dataset 1
full_1=pd.DataFrame(full_1_X_minmax)
full_1['y']=full_1_y
check_data1=full_1.describe().transpose()
full_1['y'].describe()
plt.show()

#Show interesting features of dataset 2
full_2=pd.DataFrame(full_2_X_minmax)
full_2['y']=full_2_y
#sn.heatmap(full_2.corr(), annot=True)
check_data2=full_2.describe().transpose()
full_2['y'].describe()
plt.show()

sn.pairplot(pd.DataFrame(full_1), hue="y")
sn.pairplot(pd.DataFrame(full_2), hue="y")
plt.show()

#Create a training dataset and a holdout set. The training set will be 
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(full_1_X_minmax, full_1_y, test_size=0.3, random_state=seed)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(full_2_X_minmax, full_2_y, test_size=0.3, random_state=seed)


#Create a range of percentages to use for the various learning curves
percentage = np.arange (.05, 1.0, .05)

#Examine k-fold cross validation

#Establish K folds of the data
 
chosen_cv=KFold(n_splits=10, shuffle=True, random_state=seed)

####Define models
decision_tree=tree.DecisionTreeClassifier(random_state=seed)
knn_classifier=KNeighborsClassifier()
ADA_classifier=AdaBoostClassifier(random_state=seed)
SVM_classifier=svm.SVC()
NeuralNets_classifier=MLPClassifier(solver='adam', hidden_layer_sizes=(20,), max_iter=2000)



######################################### Dataset 1      #########################################


######################################### Decision Trees #########################################

#Validation Curve for Decision Tree - Max Depth Hyperparameter
hyper_1=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plot_validation_curve(decision_tree, X_1_train, y_1_train, "max_depth",
                      hyper_1, "accuracy", chosen_cv, "Decision Tree", "Max Depth")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=[1,2,3,4,5,6,7,8,9,10]
plot_validation_curve(decision_tree, X_1_train, y_1_train, "max_features",
                      hyper_2, "accuracy", chosen_cv, "Decision Tree", "Max Number of Features")

#Validation Curve for Decision Tree - Max Leaf Nodes Hyperparameter
hyper_3=[2,3,4,5,6,7,8,9,10,15,20,25,50,100,200,1000]
plot_validation_curve(decision_tree, X_1_train, y_1_train, "max_leaf_nodes",
                      hyper_3, "accuracy", chosen_cv, "Decision Tree", "Max Leaf Nodes")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_4=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
         'max_features':[1,2,3,4,5,6,7,8,9,10],
         'max_leaf_nodes':[2,3,4,5,6,7,8,9,10,15,20,25,50,100,200,1000]}]


classifier=GridSearchCV(decision_tree, hyper_4, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_1_train, y_1_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)

max_dep=classifier.best_params_['max_depth']
max_feat=classifier.best_params_['max_features']
max_leaves=classifier.best_params_['max_leaf_nodes']
decision_tree_final_1=tree.DecisionTreeClassifier(max_depth=max_dep, max_features=max_feat, max_leaf_nodes=max_leaves,random_state=seed)
decision_tree_final_1.fit(X_1_train, y_1_train)
pred_train=decision_tree_final_1.predict(X_1_train)
pred_test=decision_tree_final_1.predict(X_1_test)

#Find the error on the full training dataset

#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Decision Tree - Training Dataset 1")

plot_conf = plot_confusion_matrix(decision_tree_final_1, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Decision Tree - Training Dataset 1")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Decision Tree - Test Dataset 1")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(decision_tree_final_1, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Decision Tree - Test Dataset 1")

plt.show()

######################################### K-Nearest Neighbors #########################################

hyperparameter_1=[{'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}]
hyperparameter_2=[{'weights' : ['uniform','distance']}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
plot_validation_curve(knn_classifier, X_1_train, y_1_train, "n_neighbors",
                      hyper_1, "accuracy", chosen_cv, "K-Nearest Neighbors", "Number of Neighbors")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=['uniform','distance']
plot_validation_curve(knn_classifier, X_1_train, y_1_train, "weights",
                      hyper_2, "accuracy", chosen_cv, "K-Nearest Neighbors", "Weight Type")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
         'weights' : ['uniform','distance']}]


classifier=GridSearchCV(knn_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_1_train, y_1_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_neighbor=classifier.best_params_['n_neighbors']
best_metric=classifier.best_params_['weights']
KNN_final_1=KNeighborsClassifier(n_neighbors=best_neighbor, weights=best_metric)
KNN_final_1.fit(X_1_train, y_1_train)
pred_train=KNN_final_1.predict(X_1_train)
pred_test=KNN_final_1.predict(X_1_test)


#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - KNN - Training Dataset 1")

plot_conf = plot_confusion_matrix(KNN_final_1, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - KNN - Training Dataset 1")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - KNN - Test Dataset 1")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(KNN_final_1, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - KNN - Test Dataset 1")

plt.show()

######################################### Adaboost #########################################

hyperparameter_1=[{'n_estimators':[10,50,100,250,500]}]
hyperparameter_2=[{'learning_rate' : [.1,.2,.5,1.0,1.5,2.0,5.0]}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=[10,50,100,250,500]
plot_validation_curve(ADA_classifier, X_1_train, y_1_train, "n_estimators",
                      hyper_1, "accuracy", chosen_cv, "ADABoost", "Number of Estimators")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=[.1,.2,.5,1.0,1.5,2.0,5.0]
plot_validation_curve(ADA_classifier, X_1_train, y_1_train, "learning_rate",
                      hyper_2, "accuracy", chosen_cv, "ADABoost", "Learning Rate")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'n_estimators':[10,50,100,250,500],
         'learning_rate' : [.1,.2,.5,1.0,1.5,2.0,5.0]}]


classifier=GridSearchCV(ADA_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_1_train, y_1_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_estimator_num=classifier.best_params_['n_estimators']
best_learning_rate=classifier.best_params_['learning_rate']
ADABoost_final_1=AdaBoostClassifier(n_estimators=best_estimator_num, learning_rate=best_learning_rate)
ADABoost_final_1.fit(X_1_train, y_1_train)
pred_train=ADABoost_final_1.predict(X_1_train)
pred_test=ADABoost_final_1.predict(X_1_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Adaboost - Training Dataset 1")

plot_conf = plot_confusion_matrix(ADABoost_final_1, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Adaboost - Training Dataset 1")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Adaboost - Test Dataset 1")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(ADABoost_final_1, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Adaboost - Test Dataset 1")

plt.show()

######################################### SVM #########################################

hyperparameter_1=[{'C': [.01, 0.1, 1, 10, 100]}]
hyperparameter_2=[{'kernel' : ['linear', 'poly', 'rbf']}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=[.01, 0.1, 1, 10, 100]
plot_validation_curve(SVM_classifier, X_1_train, y_1_train, "C",
                      hyper_1, "accuracy", chosen_cv, "SVM", "Regularization Parameter")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=['linear', 'poly', 'rbf']
plot_validation_curve(SVM_classifier, X_1_train, y_1_train, "kernel",
                      hyper_2, "accuracy", chosen_cv, "SVM", "Kernel")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'C': [.01, 0.1, 1, 10, 100],
         'kernel' : ['linear', 'poly', 'rbf']}]


classifier=GridSearchCV(SVM_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_1_train, y_1_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_C=classifier.best_params_['C']
best_kernel=classifier.best_params_['kernel']
SVM_final_1=svm.SVC(C=best_C, kernel=best_kernel)
SVM_final_1.fit(X_1_train, y_1_train)
pred_train=SVM_final_1.predict(X_1_train)
pred_test=SVM_final_1.predict(X_1_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - SVM - Training Dataset 1")

plot_conf = plot_confusion_matrix(SVM_final_1, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - SVM - Training Dataset 1")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - SVM - Test Dataset 1")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(SVM_final_1, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - SVM - Test Dataset 1")

plt.show()



######################################### Dataset 2 #########################################

######################################### Decision Trees #########################################

#Validation Curve for Decision Tree - Max Depth Hyperparameter
hyper_1=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plot_validation_curve(decision_tree, X_2_train, y_2_train, "max_depth",
                      hyper_1, "accuracy", chosen_cv, "Decision Tree", "Max Depth")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=[1,2,3,4,5,6,7,8,9,10]
plot_validation_curve(decision_tree, X_2_train, y_2_train, "max_features",
                      hyper_2, "accuracy", chosen_cv, "Decision Tree", "Max Number of Features")

#Validation Curve for Decision Tree - Max Leaf Nodes Hyperparameter
hyper_3=[2,3,4,5,6,7,8,9,10,15,20,25,50,100,200,1000]
plot_validation_curve(decision_tree, X_2_train, y_2_train, "max_leaf_nodes",
                      hyper_3, "accuracy", chosen_cv, "Decision Tree", "Max Leaf Nodes")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_4=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
         'max_features':[1,2,3,4,5,6,7,8,9,10],
         'max_leaf_nodes':[2,3,4,5,6,7,8,9,10,15,20,25,50,100,200,1000]}]


classifier=GridSearchCV(decision_tree, hyper_4, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_2_train, y_2_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


max_dep=classifier.best_params_['max_depth']
max_feat=classifier.best_params_['max_features']
max_leaves=classifier.best_params_['max_leaf_nodes']
decision_tree_final_2=tree.DecisionTreeClassifier(max_depth=max_dep, max_features=max_feat, max_leaf_nodes=max_leaves,random_state=seed)
decision_tree_final_2.fit(X_2_train, y_2_train)
pred_train=decision_tree_final_2.predict(X_2_train)
pred_test=decision_tree_final_2.predict(X_2_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_2_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Decision Tree - Training Dataset 2")

plot_conf = plot_confusion_matrix(decision_tree_final_2, X_2_train, y_2_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Decision Tree - Training Dataset 2")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_2_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Decision Tree - Test Dataset 2")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(decision_tree_final_2, X_2_test, y_2_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Decision Tree - Test Dataset 2")

plt.show()

######################################### K-Nearest Neighbors #########################################

hyperparameter_1=[{'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}]
hyperparameter_2=[{'weights' : ['uniform','distance']}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
plot_validation_curve(knn_classifier, X_2_train, y_2_train, "n_neighbors",
                      hyper_1, "accuracy", chosen_cv, "K-Nearest Neighbors", "Number of Neighbors")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=['uniform','distance']
plot_validation_curve(knn_classifier, X_2_train, y_2_train, "weights",
                      hyper_2, "accuracy", chosen_cv, "K-Nearest Neighbors", "Weight Type")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
         'weights' : ['uniform','distance']}]


classifier=GridSearchCV(knn_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_2_train, y_2_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_neighbor=classifier.best_params_['n_neighbors']
best_metric=classifier.best_params_['weights']
KNN_final_2=KNeighborsClassifier(n_neighbors=best_neighbor, weights=best_metric)
KNN_final_2.fit(X_2_train, y_2_train)
pred_train=KNN_final_2.predict(X_2_train)
pred_test=KNN_final_2.predict(X_2_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_2_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - KNN - Training Dataset 2")

plot_conf = plot_confusion_matrix(KNN_final_2, X_2_train, y_2_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - KNN- Training Dataset 2")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_2_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - KNN - Test Dataset 2")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(KNN_final_2, X_2_test, y_2_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - KNN - Test Dataset 2")

plt.show()

######################################### Adaboost #########################################

hyperparameter_1=[{'n_estimators':[10,50,100,250,500]}]
hyperparameter_2=[{'learning_rate' : [.1,.2,.5,1.0,1.5,2.0,5.0]}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=[10,50,100,250,500]
plot_validation_curve(ADA_classifier, X_2_train, y_2_train, "n_estimators",
                      hyper_1, "accuracy", chosen_cv, "ADABoost", "Number of Estimators")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=[.1,.2,.5,1.0,1.5,2.0,5.0]
plot_validation_curve(ADA_classifier, X_2_train, y_2_train, "learning_rate",
                      hyper_2, "accuracy", chosen_cv, "ADABoost", "Learning Rate")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'n_estimators':[10,50,100,250,500],
         'learning_rate' : [.1,.2,.5,1.0,1.5,2.0,5.0]}]


classifier=GridSearchCV(ADA_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_2_train, y_2_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_estimator_num=classifier.best_params_['n_estimators']
best_learning_rate=classifier.best_params_['learning_rate']
ADABoost_final_2=AdaBoostClassifier(n_estimators=best_estimator_num, learning_rate=best_learning_rate)
ADABoost_final_2.fit(X_2_train, y_2_train)
pred_train=ADABoost_final_2.predict(X_2_train)
pred_test=ADABoost_final_2.predict(X_2_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_2_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - ADABoost - Training Dataset 2")

plot_conf = plot_confusion_matrix(ADABoost_final_2, X_2_train, y_2_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - ADABoost - Training Dataset 2")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_2_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - ADABoost - Test Dataset 2")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(ADABoost_final_2, X_2_test, y_2_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - ADABoost - Test Dataset 2")

plt.show()


######################################### SVM #########################################

hyperparameter_1=[{'C': [.01, 0.1, 1, 10]}]
hyperparameter_2=[{'kernel' : ['linear', 'poly', 'rbf']}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=[.01, 0.1, 1, 10]
plot_validation_curve(SVM_classifier, X_2_train, y_2_train, "C",
                      hyper_1, "accuracy", chosen_cv, "SVM", "Regularization Parameter")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=['linear', 'poly', 'rbf']
plot_validation_curve(SVM_classifier, X_2_train, y_2_train, "kernel",
                      hyper_2, "accuracy", chosen_cv, "SVM", "Kernel")

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'C': [.01, 0.1, 1, 10],
         'kernel' : ['linear', 'poly', 'rbf']}]


classifier=GridSearchCV(SVM_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_2_train, y_2_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_C=classifier.best_params_['C']
best_kernel=classifier.best_params_['kernel']
SVM_final_2=svm.SVC(C=best_C, kernel=best_kernel)
SVM_final_2.fit(X_2_train, y_2_train)
pred_train=SVM_final_2.predict(X_2_train)
pred_test=SVM_final_2.predict(X_2_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_2_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - SVM - Training Dataset 2")

plot_conf = plot_confusion_matrix(SVM_final_2, X_2_train, y_2_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - SVM - Training Dataset 2")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_2_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - SVM - Test Dataset 2")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(SVM_final_2, X_2_test, y_2_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - SVM - Test Dataset 2")

plt.show()


######################################### Neural Networks #########################################

#Dataset 1

NeuralNets_classifier=MLPClassifier(solver='adam', max_iter=2000, learning_rate_init=.001)

hyperparameter_1=[{'activation': ['identity', 'logistic', 'tanh', 'relu']}]
hyperparameter_2=[{'hidden_layer_sizes': [(5), (5,5), (5,5,5)]}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=['identity', 'logistic', 'tanh', 'relu']
plot_validation_curve(NeuralNets_classifier, X_1_train, y_1_train, "activation",
                      hyper_1, "accuracy", chosen_cv, "Neural Networks", "Activation Function")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=[(5), (5,5), (5,5,5)]

train_scores, test_scores = validation_curve(NeuralNets_classifier, X_1_train, y_1_train, param_name="hidden_layer_sizes",
                                             param_range=hyper_2, scoring="accuracy", n_jobs=-1, cv=chosen_cv)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
    
ind=np.argmax(np.mean(test_scores, axis=1), axis=0)

hyper=["1","2","3"]
plt.title("Validation Curve with Neural Networks")
plt.xlabel("# of Hidden Layers")
plt.ylabel("Score - Accuracy")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(hyper, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(hyper, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(hyper, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(hyper, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

plt.annotate("Best Value: {0}".format(hyper[ind]),
        xy=(hyper[ind],test_scores_mean[ind]), xycoords='data',
        xytext=(20, -30), textcoords='offset points',
        arrowprops=dict(facecolor='black', shrink=0.05),
        horizontalalignment='right', verticalalignment='bottom')
plt.show()

#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'activation': ['identity', 'logistic', 'tanh', 'relu'],
         'hidden_layer_sizes': [(5), (5,5), (5,5,5)]}]


classifier=GridSearchCV(NeuralNets_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_1_train, y_1_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)


best_activation=classifier.best_params_['activation']
best_num_hidden_layers=classifier.best_params_['hidden_layer_sizes']

NeuralNets_final_1=MLPClassifier(activation=best_activation, hidden_layer_sizes=best_num_hidden_layers, max_iter=2000, solver='adam', learning_rate_init=.001)

NN=NeuralNets_final_1.fit(X_1_train, y_1_train)
pred_train=NeuralNets_final_1.predict(X_1_train)
pred_test=NeuralNets_final_1.predict(X_1_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Training Dataset 1")

plot_conf = plot_confusion_matrix(NeuralNets_final_1, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Training Dataset 1")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - KNN - Test Dataset 1")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(NeuralNets_final_1, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Test Dataset 1")

plt.show()

#Dataset 2

hyperparameter_1=[{'activation': ['identity', 'logistic', 'tanh', 'relu']}]
hyperparameter_2=[{'hidden_layer_sizes': [(5), (5,5), (5,5,5)]}]

#Validation Curve for KNN - # of Neighbors Hyperparameter
hyper_1=['identity', 'logistic', 'tanh', 'relu']
plot_validation_curve(NeuralNets_classifier, X_2_train, y_2_train, "activation",
                      hyper_1, "accuracy", chosen_cv, "Neural Networks", "Activation Function")

#Validation Curve for Decision Tree - Max Features Hyperparameter
hyper_2=[(5), (5,5), (5,5,5)]

train_scores, test_scores = validation_curve(NeuralNets_classifier, X_2_train, y_2_train, param_name="hidden_layer_sizes",
                                             param_range=hyper_2, scoring="accuracy", n_jobs=-1, cv=chosen_cv)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
    
ind=np.argmax(np.mean(test_scores, axis=1), axis=0)

hyper=["1","2","3"]
plt.title("Validation Curve with Neural Networks")
plt.xlabel("# of Hidden Layers")
plt.ylabel("Score - Accuracy")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(hyper, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(hyper, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(hyper, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(hyper, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

plt.annotate("Best Value: {0}".format(hyper[ind]),
        xy=(hyper[ind],test_scores_mean[ind]), xycoords='data',
        xytext=(20, -30), textcoords='offset points',
        arrowprops=dict(facecolor='black', shrink=0.05),
        horizontalalignment='right', verticalalignment='bottom')
plt.show()


#After testing the two hyperparameters I will run a GridSearch to find the best combination of both
hyper_3=[{'activation': ['identity', 'logistic', 'tanh', 'relu'],
         'hidden_layer_sizes': [(5), (5,5), (5,5,5)]}]


classifier=GridSearchCV(NeuralNets_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
classifier.fit(X_2_train, y_2_train)
print('Highest Score: %s' % classifier.best_score_)
print('Corresponding Hyperparameters: %s' % classifier.best_params_)

best_activation=classifier.best_params_['activation']
best_num_hidden_layers=classifier.best_params_['hidden_layer_sizes']

NeuralNets_final_2=MLPClassifier(activation=best_activation, hidden_layer_sizes=best_num_hidden_layers, max_iter=2000, solver='adam', learning_rate_init=.001)
NN=NeuralNets_final_2.fit(X_2_train, y_2_train)

pred_train=NeuralNets_final_2.predict(X_2_train)
pred_test=NeuralNets_final_2.predict(X_2_test)

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_2_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Training Dataset 2")

plot_conf = plot_confusion_matrix(NeuralNets_final_2, X_2_train, y_2_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Training Dataset 2")

plt.show()

#Find the error on the test dataset
ax = plt.axes()
clf_report = classification_report(y_2_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Test Dataset 2")

#Create a plot of the confusion matrix for the test dataset
plot_conf = plot_confusion_matrix(NeuralNets_final_2, X_2_test, y_2_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Test Dataset 2")

plt.show()



###Learning Curves###

#Plot Learning Curves for both datasets and compare
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(decision_tree_final_1, "Learning Curves: Decision Tree - Dataset 1", X_1_train, y_1_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plot_learning_curve(decision_tree_final_2, "Learning Curves: Decision Tree - Dataset 2", X_2_train, y_2_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plt.show()

#Plot Learning Curves for both datasets and compare
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(KNN_final_1, "Learning Curves: KNN - Dataset 1", X_1_train, y_1_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plot_learning_curve(KNN_final_2, "Learning Curves: KNN - Dataset 2", X_2_train, y_2_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plt.show()

#Plot Learning Curves for both datasets and compare
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(ADABoost_final_1, "Learning Curves: Adaboost - Dataset 1", X_1_train, y_1_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plot_learning_curve(ADABoost_final_2, "Learning Curves: Adaboost - Dataset 2", X_2_train, y_2_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plt.show()

#Plot Learning Curves for both datasets and compare
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(SVM_final_1, "Learning Curves: SVM - Dataset 1", X_1_train, y_1_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plot_learning_curve(SVM_final_2, "Learning Curves: SVM - Dataset 2", X_2_train, y_2_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plt.show()

#Plot Learning Curves for both datasets and compare
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(NeuralNets_final_1, "Learning Curves: Neural Nets - Dataset 1", X_1_train, y_1_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plot_learning_curve(NeuralNets_final_2, "Learning Curves: Neural Nets - Dataset 2", X_2_train, y_2_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=chosen_cv, n_jobs=-1, train_sizes=percentage)
plt.show()

#Neural Network Learning Curve - Iterations
NeuralNets_final_1=MLPClassifier(activation='tanh', hidden_layer_sizes=(5,5,5), max_iter=2000, solver='adam', learning_rate_init=.001, validation_fraction=.3, early_stopping=True, random_state=seed)
NN=NeuralNets_final_1.fit(X_1_train, y_1_train)

plt.title("Learning Curve - Neural Network - Dataset 1")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(NN.loss_curve_, label="Training")
Loss=[]
for i in NN.validation_scores_:
    Loss.append(1-i)
plt.plot(Loss, label="Validation")
plt.legend(loc="best")
plt.show()

NeuralNets_final_2=MLPClassifier(activation='tanh', hidden_layer_sizes=(5,5), max_iter=2000, solver='adam', learning_rate_init=.001, validation_fraction=.3, early_stopping=True, random_state=seed)
NN=NeuralNets_final_2.fit(X_2_train, y_2_train)

plt.title("Learning Curve - Neural Network - Dataset 2")
plt.xlabel("Iteration")
plt.ylabel("Loss")
Loss=[]
for i in NN.validation_scores_:
    Loss.append(1-i)
plt.plot(NN.loss_curve_, label="Training")
plt.plot(Loss, label="Validation")
plt.legend(loc="best")
plt.show()

print(decision_tree_final_1)
print(decision_tree_final_2)

print(KNN_final_1)
print(KNN_final_2)

print(ADABoost_final_1)
print(ADABoost_final_2)

print(SVM_final_1)
print(SVM_final_2)

print(NeuralNets_final_1)
print(NeuralNets_final_2)