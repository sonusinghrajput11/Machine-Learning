
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


dataset = pd.read_csv('../ML_dataset/salary_exp/Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X_Train = X[:20]
X_Test = X[20:] 
Y_Train = Y[:20] 
Y_Test = Y[20:]


# In[ ]:


no_of_epochs = 1000
learning_rate = 0.05
display_step = 10
training_loss = 0.0

#Defining Weight and bais
W = np.zeros((X_Train.shape[1], Y_Train.shape[1]), dtype=np.float);
b = np.zeros((1,1), dtype=np.float);

X_Train_T = np.transpose(X_Train)
no_of_sample = X_Train.shape[0]


# In[ ]:


def calculate_loss(X_Input, Y_Input):
    Y_Pred = np.dot(X_Input, W) + b
    
    diff = (Y_Pred - Y_Input)
    
    temp_loss = (np.square(diff)).sum()
    
    return temp_loss


# In[ ]:


for epoch in range(no_of_epochs):
    
    Y_Prediction = np.dot(X_Train, W) + b
    
    diff = Y_Prediction - Y_Train
    
    gradient = np.dot(X_Train_T, diff) / no_of_sample
    
    W = W - (learning_rate * (gradient))
    
    b = b - (learning_rate * (np.sum(diff, keepdims=True)) / no_of_sample)   
    
    if (epoch + 1) % display_step == 0:
        temp_loss = (np.square(diff)).sum()
        training_loss = training_loss + temp_loss
        print("Epoch:", '%d' % (epoch + 1), "loss=", "{:.9f}".format(temp_loss))
        
print("***********************************Training Finished!******************************************")
print("Training loss=", training_loss, "W=", W, "b=", b, '\n')


# In[ ]:


def predict(X_Input):
    return np.dot(X_Input, W) + b


# In[ ]:


# Visualising the Training set results
plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, predict(X_Train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Test, predict(X_Test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 

