import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

#Reading boston dataset
X_Actual, Y_Actual = load_boston(True)

#Taking fisrt 300 samples for training the model
X_Train = scale(X_Actual[:300])
Y_Train = Y_Actual[:300]

#Taking samples from 300 to 400 for validatiung the model
X_Validation = scale(X_Actual[300:400])
Y_Validation = Y_Actual[300:400]

#Taking data 400 onwards for testung the model 
X_Test = scale(X_Actual[400:])
Y_Test = Y_Actual[400:]

#Definig place holder to hold data
X = tf.placeholder(tf.float32, shape=(None, 13))
Y = tf.placeholder(tf.float32, shape=(None))

#Defining Weight and bais
W = tf.Variable(tf.random_normal([13,1], stddev = 0.35), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bais")

no_of_samples = 300

#Caluclating squared error
Y_Prediction = tf.add(tf.matmul(X, W), b)
square_error = tf.pow(tf.subtract(Y, Y_Prediction), 2)

#Defining loss function
loss = tf.reduce_mean(square_error / (2 * no_of_samples))

#Learing rate taken as 0.01
learning_rate = 0.01

#Defining garient decent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#Trainig the model : - Printing loss in the interval of 1000 epochs
no_of_epochs = 50000
display_step = 1000
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(no_of_epochs):
    sess.run(optimizer, feed_dict = {X : X_Train, Y : Y_Train})
    if (epoch + 1) % display_step == 0:
        c = sess.run(loss, feed_dict = {X: X_Train, Y: Y_Train})
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c))
        
print("***********************************Training Finished!******************************************")
training_loss = sess.run(loss, feed_dict = {X: X_Train, Y: Y_Train})
print("Training loss=", training_loss, "W=", sess.run(W), "b=", sess.run(b), '\n')

#Model Validation and Testing
validation_loss = sess.run(loss, feed_dict = {X: X_Validation, Y: Y_Validation})
print("Validation loss=", validation_loss, '\n')

test_loss = sess.run(loss, feed_dict = {X: X_Test, Y: Y_Test})
print("Test loss=", test_loss, '\n')

correct_prediction = tf.equal(Y_Prediction, Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy =", sess.run(accuracy, feed_dict={X: X_Test, Y: Y_Test}))

prediction = Y_Prediction
print("Predictions =", prediction.eval(feed_dict={X: X_Test}, session=sess))
