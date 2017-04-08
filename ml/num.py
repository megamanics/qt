#!/usr/bin/env python

"""
Numerai data.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import metrics, preprocessing, linear_model,lda,naive_bayes,qda,ensemble,svm,neural_network
from sklearn.decomposition import FastICA, PCA, NMF  
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import VotingClassifier  
from sklearn.neural_network import BernoulliRBM  
from sklearn.pipeline import Pipeline  
from sklearn.calibration import CalibratedClassifierCV  
from sklearn.cluster import MiniBatchKMeans

models = [("ADA", sk.ensemble.AdaBoostClassifier()),
          #("ABR", sk.ensemble.AdaBoostRegressor()),
          ("EBC", sk.ensemble.BaggingClassifier()),
          ("RFC", sk.ensemble.RandomForestClassifier()),
          #("NNB", sk.neural_network.BernoulliRBM()),
          ("LDA", sk.lda.LDA()), 
          ("QDA", sk.qda.QDA()),
          ("LoR", sk.linear_model.LogisticRegression()),
          #("LIR", sk.linear_model.LinearRegression()),
          #("ARD", sk.linear_model.ARDRegression()),
          #("SGD", sk.linear_model.SGDRegressor()),
          #("BAR", sk.linear_model.BayesianRidge()),
          #("PAR", sk.linear_model.PassiveAggressiveRegressor()),
          #("ELN", sk.linear_model.ElasticNet()),
          #("LAL", sk.linear_model.LassoLars()),
          #("LAR", sk.linear_model.Lars()),
          #("LAS", sk.linear_model.Lasso()),
          #("LAC", sk.linear_model.LassoCV()),
          #("LLC", sk.linear_model.LassoLarsCV()),         
          ("GNB", sk.naive_bayes.GaussianNB()),
          ("BNB", sk.naive_bayes.BernoulliNB()),
          #("SVC", sk.svm.LinearSVC()),
          ("SVR", sk.svm.LinearSVR())
         ]
def info():
    import numerapi

    napi = numerapi.NumerAPI('email', 'password')
    napi.download_current_dataset(dest_path='.', unzip=True)
    napi.upload_prediction('path/to/prediction.csv')
    napi.get_user('username')
    napi.get_scores('username')
    napi.get_earnings_per_round('username')
    
def loaddata():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data   = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)
    
    # Transform the loaded CSV data into numpy arrays
    y    = training_data['target']
    x    = training_data.drop('target', axis=1)
    tid  = prediction_data['t_id']
    test = prediction_data.drop('t_id', axis=1)
    
    return tid,x,y,test


def regress(name,model,tid,x,y,test):
    # Your model is trained on the numerai_training_data
    model.fit(x, y)
  
    ytrainpred = model.predict(x)
    


    correct = (ytrainpred == y).sum()
    print name,"accuracy:",np.float(correct)*100/len(x)
    
    ytrainprob = model.predict_proba(x)
    print name,"mean probability",(ytrainprob[:,1]).mean()
    
    logloss = metrics.log_loss(y, ytrainprob[:,1])
    print "logloss:",logloss
        
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(test)
    results      = y_prediction[:, 1]
    savedata(name,results,tid)
    
def random(tid,test):
    """This is the tricky part that requires some knowledge of the 
    scoring mechanism based on log loss function:
    LogLoss Note that a perfect prediction has log loss 0. 
    The log loss function penalizes confident but wrong estimates of the probability. 
    Since our estimates are essentially random, if we upload the averages file  
    it will score near the bottom because it contains many wrong but confident 
    values of the estimated probability in the range of about 0.1 to about 0.8. 
    We will scale the estimated probabilities derived via averaging the values of 
    the features for each row to lie between 0.49 and 0.51.
        avg1.describe()
        count    264877.000000
        mean          0.502689
        std           0.021366
        min           0.425658
        25%           0.487593
        50%           0.502576
        75%           0.517588
        max           0.581305
        prop.describe()
        count    264877.000000
        mean          0.500752
        std           0.000580
        min           0.498661
        25%           0.500342
        50%           0.500749
        75%           0.501157
        max           0.502887
        """
    avg1 = test.apply(np.average,axis=1)
    #((0.51-0.49)*(B2-0.106729402)/(0.843236581-0.106729402))+0.49
    prop = ((0.51-0.49)*(avg1-0.106729402)/(0.843236581-0.106729402))+0.49
    savedata("ran",prop,tid)    
    
def savedata(name,results,tid):
    results_df = pd.DataFrame(data={'probability':results})
    print name,results_df.describe()    
    joined = pd.DataFrame(tid).join(results_df)
    # Save the predictions out to a CSV file
    joined.to_csv(name+".csv", index=False)
    # Now you can upload these predictions on numer.ai
    
    
def main():

    tid,x,y,test=loaddata()
    #model = linear_model.LogisticRegression(n_jobs=-1)
    #regress("LR",model,tid,x,y,test)
    #random(tid,test)
    
    for m in models:
        print m[0]
        regress(m[0],m[1],tid,x,y,test)

    training = x#df.values[:, :21]  
    classes =  y#df.values[:, -1]  
    training = preprocessing.scale(training)  
    kmeans = MiniBatchKMeans(n_clusters=500, init_size=6000).fit(training)  
    labels = kmeans.predict(training)
    
    clusters = {}  
    for i in range(0, np.shape(training)[0]):  
        label = labels[i]  
        if label not in clusters:  
            clusters[label] = training[i, :]  
        else:  
            clusters[label] = np.vstack((clusters[label], training[i, :]))
    
    params = {'n_estimators': 100, 'max_depth': 3, 'subsample': 0.5,  
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}  
    gbm = GradientBoostingClassifier(**params)  
    ica = FastICA(10,max_iter=100000, tol=0.16)
    
    icas = {}  
    for label in clusters:  
        icas[label] = ica.fit(clusters[label])
    
    factors = np.zeros((np.shape(training)[0], 10))
    
    for i in range(0, np.shape(training)[0]):  
        factors[i, :] = icas[labels[i]].transform(training[i, :].reshape(1, -1))
    
    gbm = gbm.fit(factors, classes)    
    
    tf = pd.read_csv('numerai_tournament_data.csv', header=0)
    forecast = tf.values[:, 1:]  
    forecast = preprocessing.scale(forecast)  
    labels = kmeans.predict(forecast)
    
    factors = np.zeros((np.shape(labels)[0], 10))  
    for i, label in enumerate(labels):  
        factors[i, :] = icas[label].transform(forecast[i, :].reshape(1, -1))
    
    proba = gbm.predict_proba(factors)
    of = pd.Series(proba[:, 1], index=tid)
    print "stats:",of.describe()    
    of.to_csv("predictionsice.csv", header=['probability'], index_label='t_id')  
    
if __name__ == '__main__':
    main()
