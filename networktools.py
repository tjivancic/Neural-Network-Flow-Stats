import sklearn.neural_network as nn
import pandas as pd
import random
import getflow as gf
import getbasin as gb
import networktools as nt
import flowstats as fs
import matplotlib.pyplot as plt
import numpy as np

def normalizecolumns(data):
    """
    Takes a vector or a pandas Dataframe and subtracts the mean and divides by 
    the range of each column 
    """
    normdat = (data-data.mean())/(data.max()-data.min())
    return normdat

def try_model(YX, indexshuff, Ntrain=3900, Nval=1300, hls = (200, 100, 50),
        plot=False):
    """
    trains and runs a MLP regression model given the parameters: Ntrain, size of
    training set, N val, size of validation set, and hls, the size of each 
    perceptron layer. If plot, plots the validation hypotheses vs input Ys.
    """
    Xsub = YX.keys()[1:]
    Ysub = YX.keys()[0:1]
    YXtrain = YX.loc[indexshuff[:Ntrain]]
    YXval = YX.loc[indexshuff[Ntrain:(Ntrain+Nval)]]
    YXtest= YX.loc[indexshuff[(Ntrain+Nval):]]
    Xtrain = YXtrain[Xsub]
    Ytrain = YXtrain[Ysub]

    Xval = YXval[Xsub]
    Yval = YXval[Ysub]

    Xtest = YXtest[Xsub]
    Ytest = YXtest[Ysub]
    
    clf = nn.MLPRegressor(hidden_layer_sizes = hls, max_iter=2000)
    clf.fit(Xtrain,Ytrain)
    if plot:
        plt.plot(np.log(Yval),np.log(clf.predict(Xval)),'ro')
        plt.show()
    return [clf.score(Xval,Yval),clf]

def test_model(YX, indexshuff, Ntrain, Nval, clf, plot=True):
    """
    given the same YX, indexshuff, Ntrain, Nval as try_model
    runs a MLP model on the test data and prints the scores for both validation 
    test data and plots the test data y values vs hypothesis 
    """
    YXval = YX.loc[indexshuff[Ntrain:(Ntrain+Nval)]]
    YXtest= YX.loc[indexshuff[(Ntrain+Nval):]]
    
    Xsub = YX.keys()[1:]
    Ysub = YX.keys()[0:1]

    Xval = YXval[Xsub]
    Yval = YXval[Ysub]

    Xtest = YXtest[Xsub]
    Ytest = YXtest[Ysub]
    
    print 'validation score:' + str(clf.score(Xval,Yval))
    print 'test score:' + str(clf.score(Xtest,Ytest))
    if plot:
        plt.plot(np.log(Ytest),np.log(clf.predict(Xtest)),'ro')
        plt.show()    

def bestof(N, modelfunc, *args):
    """
    with so many variables the method often finds local maxima, best of repeats 
    the analysis N times and returns the model with the best validation score
    """
    model_scores = [[] for i in range(N)]
    for i in range(N):
        model_scores[i] = modelfunc(*args)
    index_min=np.array([a[0] for a in model_scores]).argmin()
    return model_scores[index_min]

    
    
    
    
    
    
    
