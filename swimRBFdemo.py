# Copyright 2018, Colin Bellinger, All rights reserved.
# paper "Synthetic oversampling with the majority class: A new perspective on handling extreme imbalance".
# IEEE 2018 International Converence on Data Mining 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random
from imblearn.over_sampling import SMOTE
import SWIM_RBF.Swim_RBF as Swim
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score

random.seed(a=125324)
np.random.seed(seed=125324)

i=0
h = .02  # step size in the mesh
MIN_SIZE = 10
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

linearly_separable = (X, y)
centers = [[1, 1], [-1, -1], [0, -0.15]]
datasets = [make_moons(n_samples=500,noise=0.35, random_state=0),
            make_circles(n_samples=500,noise=0.25, factor=0.5, random_state=1),
            make_classification(n_samples=500,n_features=2,n_redundant=0,
                             n_clusters_per_class=2, n_classes=2),
            make_classification(n_samples=500,n_features=2,n_redundant=0,n_informative=2,
                             n_clusters_per_class=2, n_classes=2),
            make_classification(n_samples=500,n_features=3,n_redundant=0,n_informative=3,
                             n_clusters_per_class=3, n_classes=2),
            make_blobs(n_samples=500, centers=centers,
                             cluster_std=[0.5, 0.25, 0.5]),
            linearly_separable
            ]
ds_names = ['moons', 'circles','clusters', 'clusters2','cluster3','blobs']
# for dsNum in [0,1,2,3,4]:
for dsNum in [5,4,3,2,1,0]:
    ds_name = ds_names[dsNum]+"1"
    print(ds_name)
    # for name, clf in zip(names, classifiers):
    name = names[2]
    clf  = classifiers[2]
    # preprocess dataset, split into training and test part
    X, y = datasets[dsNum]
    # X = np.concatenate([X,X],axis=1)
    if ds_name == 'cluster31':
        X = X[:, 0:-1]
    if ds_name == 'blobs1':
        y[np.where(y==0)[0]] = 1
        y[np.where(y==2)[0]] = 0
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # just plot the dataset first
    # i=1
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    x_train_maj = X_train[np.where(y_train==0)[0], :]
    x_train_min = X_train[np.random.choice((np.where(y_train==1)[0]),MIN_SIZE), :]
    x_trainImb = np.append(x_train_maj, x_train_min,axis=0)
    y_trainImb = np.append(np.zeros(x_train_maj.shape[0]), np.ones(MIN_SIZE))
    #
    ax = plt.subplot(1,1, 1)
    clf.fit(x_trainImb, y_trainImb)
    y_pred = clf.predict(X_test)
    #
    g_mean = 1.0
    #
    for label in np.unique(y_test):
        idx = (y_test == label)
        g_mean *= accuracy_score(y_test[idx], y_pred[idx])
    #
    g_mean = np.sqrt(g_mean)
    score = g_mean
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    # Plot also the training points
    # and testing points
    # ax.scatter(X_test[np.where(y_test==0)[0], 0], X_test[np.where(y_test==0)[0], 1], color="red", cmap=cm_bright,
    #            edgecolors='black',alpha=0.5,label="Majority Testing")
    # ax.scatter(X_test[np.where(y_test==1)[0], 0], X_test[np.where(y_test==1)[0], 1], color="blue", cmap=cm_bright,
    #            edgecolors='black',alpha=0.5,label="Minority Testing")
    ax.scatter(x_trainImb[np.where(y_trainImb==0)[0], 0], x_trainImb[np.where(y_trainImb==0)[0], 1], color="red", cmap=cm_bright,
               edgecolors='w', marker='s',alpha=0.85,label="Majority Training")
    ax.scatter(x_trainImb[np.where(y_trainImb==1)[0], 0], x_trainImb[np.where(y_trainImb==1)[0], 1], color="blue", cmap=cm_bright,
               edgecolors='w', marker='s',alpha=0.85,label="Minority Training")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.legend()
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    # plt.show()
    plt.savefig("demoResults/rbfTest/imbalanced_"+ds_name+"MinSize"+str(MIN_SIZE)+"_raw"+name+".pdf")
    plt.close()
    for ep in [None, 0.25, 1,1.5,2,3]:
        print(ep)
        ax = plt.subplot(1,1, 1)
        numSamples = np.sum(y_trainImb==0)-np.sum(y_trainImb==1)
        erbo = Swim.SwimRBF(epsilon=ep)
        np.random.seed(seed=1234)
        X_res, y_res = erbo.extremeRBOSample(x_trainImb, y_trainImb, numSamples)

        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_test)
        g_mean = 1.0
        #
        for label in np.unique(y_test):
            idx = (y_test == label)
            g_mean *= accuracy_score(y_test[idx], y_pred[idx])
        #
        g_mean = np.sqrt(g_mean)
        score = g_mean
        #
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        #
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # Plot also the training points
        # and testing points
        # ax.scatter(X_test[np.where(y_test==1)[0], 0], X_test[np.where(y_test==1)[0], 1], color="blue", cmap=cm_bright,
        #            edgecolors='black',alpha=0.5,label="Minority Testing")
        # ax.scatter(X_test[np.where(y_test==0)[0], 0], X_test[np.where(y_test==0)[0], 1], color="red", cmap=cm_bright,
        #            edgecolors='black',alpha=0.5,label="Majority Testing")

        ax.scatter(x_trainImb[np.where(y_trainImb==0)[0], 0], x_trainImb[np.where(y_trainImb==0)[0], 1], color="red", cmap=cm_bright,
                   edgecolors='w', marker='s',alpha=0.85,label="Majority Training")

        ax.scatter(X_res[np.where(y_res==1)[0], 0], X_res[np.where(y_res==1)[0], 1], color="blue", cmap=cm_bright,
                   edgecolors='k', marker='s', alpha=0.75,label="Synthetic Training")

        ax.scatter(x_trainImb[np.where(y_trainImb==1)[0], 0], x_trainImb[np.where(y_trainImb==1)[0], 1], color="blue", cmap=cm_bright,
                   edgecolors='w', marker='s',alpha=0.75,label="Minority Training")
        ax.set_title("Epsilon = " +str(erbo.epsilon))
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.legend()
        ax.text(xx.max() - .3, yy.min() + .3, ('0%0.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        # plt.show()
        plt.savefig("demoResults/rbfTest/imbalanced_"+ds_name+"MinSize"+str(MIN_SIZE)+"_SwimRBF_Epsilon"+str(ep)+name+".pdf")
        plt.close()

    # SMOTE
    for k in [5,7,9]:
        ax = plt.subplot(1,1, 1)
        sm = SMOTE(k_neighbors=k)
        np.random.seed(seed=1234)
        X_res, y_res = sm.fit_sample(x_trainImb, y_trainImb)
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_test)
        g_mean = 1.0
        #
        for label in np.unique(y_test):
            idx = (y_test == label)
            g_mean *= accuracy_score(y_test[idx], y_pred[idx])
        #
        g_mean = np.sqrt(g_mean)
        score = g_mean
        #
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        #
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # Plot also the training points
        # and testing points
        # ax.scatter(X_test[np.where(y_test==1)[0], 0], X_test[np.where(y_test==1)[0], 1], color="blue", cmap=cm_bright,
        #            edgecolors='black',alpha=0.5,label="Minority Testing")
        # ax.scatter(X_test[np.where(y_test==0)[0], 0], X_test[np.where(y_test==0)[0], 1], color="red", cmap=cm_bright,
        #            edgecolors='black',alpha=0.5,label="Majority Testing")

        ax.scatter(x_trainImb[np.where(y_trainImb==0)[0], 0], x_trainImb[np.where(y_trainImb==0)[0], 1], color="red", cmap=cm_bright,
                   edgecolors='w', marker='s',alpha=0.85,label="Majority Training")

        ax.scatter(X_res[np.where(y_res==1)[0], 0], X_res[np.where(y_res==1)[0], 1], color="blue", cmap=cm_bright,
                   edgecolors='k', marker='s', alpha=0.75,label="Synthetic Training")

        ax.scatter(x_trainImb[np.where(y_trainImb==1)[0], 0], x_trainImb[np.where(y_trainImb==1)[0], 1], color="blue", cmap=cm_bright,
                   edgecolors='w', marker='s',alpha=0.75,label="Minority Training")
        ax.set_title("K = " +str(k))
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.legend()
        ax.text(xx.max() - .3, yy.min() + .3, ('0%0.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        # plt.show()
        plt.savefig("demoResults/rbfTest/imbalanced_"+ds_name+"MinSize"+str(MIN_SIZE)+"_SmoteK"+str(k)+name+".pdf")
        plt.close()
