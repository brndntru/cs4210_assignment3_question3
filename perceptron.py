#-------------------------------------------------------------------------
# AUTHOR: Brandon Trieu
# FILENAME: perceptron.py
# SPECIFICATION: This program implements a Single Layer Perceptron and Multi-Layer Perceptron (MLP) classify optically recognized handwritten digits
#                using training samples from optdigits.tra and optdigits.tes. It uses different combinations of learning rates and shuffle settings to
#                train the models and computes their accuracy, updating and printing the accuracy of each classifier when it gets higher, togehter with
#                their hyperparameters.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('assignment3/optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('assignment3/optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

best_perceptron_acc = 0.0
best_mlp_acc = 0.0

for learning_rate in n: #iterates over n

    for shuffle_flag in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for algorithm in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here
            if algorithm == 'Perceptron':
                clf = Perceptron(eta0 = learning_rate, shuffle = shuffle_flag, max_iter = 1000, tol = None)
            else:
                clf = MLPClassifier(activation = 'logistic', learning_rate_init = learning_rate, hidden_layer_sizes = (25,), shuffle = shuffle_flag, max_iter = 1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            predictions = clf.predict(X_test)
            accuracy = np.mean(predictions == y_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            if algorithm == 'Perceptron' and accuracy > best_perceptron_acc:
                best_perceptron_acc = accuracy
                print("Highest Perceptron accuracy so far: {:.4f}, Parameters: learning rate = {}, shuffle = {}".format(
                    best_perceptron_acc, learning_rate, shuffle_flag))
            elif algorithm == 'MLP' and accuracy > best_mlp_acc:
                best_mlp_acc = accuracy
                print("Highest MLP accuracy so far: {:.4f}, Parameters: learning rate = {}, shuffle = {}".format(
                    best_mlp_acc, learning_rate, shuffle_flag))