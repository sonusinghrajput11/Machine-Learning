
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns

#Reading breast_cancer dataset
X_Actual, Y_Actual = load_breast_cancer(True)

#Taking fisrt 500 samples for training the model
X_Train = scale(X_Actual[:500])
Y_Train = Y_Actual[:500].reshape(-1,1)

#Taking data 500 onwards for testung the model 
X_Test = scale(X_Actual[500:])
Y_Test = Y_Actual[500:].reshape(-1,1)

no_of_epochs = 5000
learning_rate = 0.05
display_step = 10
training_loss = 0.0

#Defining Weight and bais
W = np.zeros((X_Train.shape[1], Y_Train.shape[1]), dtype=np.float);
b = np.zeros((1,1), dtype=np.float);

X_Train_T = np.transpose(X_Train)
no_of_sample = X_Train.shape[0]

def calculate_loss(X_Input, Y_Input):
    Y_Pred = np.dot(X_Input, W) + b
    
    diff = (Y_Pred - Y_Input)
    
    temp_loss = (np.square(diff)).sum()
    
    return temp_loss

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

for epoch in range(no_of_epochs):
    
    Y_Temp = np.dot(X_Train, W) + b
    
    Y_Prediction = sigmoid_activation(Y_Temp)
        
    diff = Y_Prediction - Y_Train
    
    gradient = np.dot(X_Train_T, diff) / no_of_sample
    
    W = W - (learning_rate * gradient)
    
    b = b - (learning_rate * (np.sum(diff, keepdims=True)) / no_of_sample)  
      
    if (epoch + 1) % display_step == 0:
        temp_loss = (np.square(diff)).sum()
        training_loss = training_loss + temp_loss
        print("Epoch:", '%d' % (epoch + 1), "loss=", "{:.9f}".format(temp_loss))
        
print("***********************************Training Finished!******************************************")
print("Training loss=", training_loss, "W=", W, "b=", b, '\n')

def check_accuracy(X_T, Y_T):
    prediction = sigmoid_activation(np.dot(X_T, W) + b)
    prediction = (prediction > 0.5).astype(int)
    prediction = np.equal(prediction, Y_T)
    prediction = np.count_nonzero(prediction == True)
    prediction = (prediction / Y_T.shape[0]) * 100
    return prediction

#Model Testing
test_loss = calculate_loss(X_Test, Y_Test)
print("Test loss=", test_loss, '\n')
print("Accuracy(%) = ",check_accuracy(X_Test, Y_Test))

