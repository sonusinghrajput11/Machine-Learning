import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

#Reading boston dataset
X_Actual, Y_Actual = load_boston(True)

#Taking fisrt 300 samples for training the model
X_Train = scale(X_Actual[:400])
Y_Train = Y_Actual[:400]

#Taking data 400 onwards for testung the model 
X_Test = scale(X_Actual[400:])
Y_Test = Y_Actual[400:]

no_of_epochs = 1000
learning_rate = 0.01
display_step = 10
training_loss = 0.0

#Defining Weight and bais
W = np.zeros((13, 1), dtype=np.float);
b = np.zeros((1,1), dtype=np.float);

X_Train_T = np.transpose(X_Train)
no_of_sample = X_Train.shape[0]

#print("X_Train ",  X_Train, "Y_Train ", Y_Train)

def calculate_loss(X_Input, Y_Input):
    Y_Pred = np.dot(X_Input, W) + b
    
    diff = (Y_Pred - Y_Input)
    
    temp_loss = (np.square(diff)).sum()
    
    return temp_loss

for epoch in range(no_of_epochs):
    
    Y_Prediction = np.dot(X_Train, W) + b
    
    diff = Y_Prediction - Y_Train
    
    gradient = np.dot(X_Train_T, diff) / no_of_sample
    
    W = W - (learning_rate * (gradient)
    
    b = b - (learning_rate * (np.sum(diff, keepdims=True)) / no_of_sample)   
    
    if (epoch + 1) % display_step == 0:
        temp_loss = (np.square(diff)).sum()
        training_loss = training_loss + temp_loss
        print("Epoch:", '%d' % (epoch + 1), "loss=", "{:.9f}".format(temp_loss))
        
print("***********************************Training Finished!******************************************")
print("Training loss=", training_loss, "W=", W, "b=", b, '\n')

#Model Validation and Testing
test_loss = calculate_loss(X_Test, Y_Test)
print("Test loss=", test_loss, '\n')

prediction = np.dot(X_Test, W) + b
print("Predictions =", prediction)
print("Actual =", Y_Test)   

