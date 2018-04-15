
# coding: utf-8

# In[102]:


import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale


# In[103]:


#Reading iris dataset
X_Actual, Y_Actual = load_iris(True)

X_Actual = (tf.random_shuffle(X_Actual)).eval()
Y_Actual = (tf.random_shuffle(Y_Actual)).eval()

#Taking fisrt 135 samples for training the model
X_Train = scale(X_Actual[:135])
Y_Train = tf.one_hot(Y_Actual[:135], 3)

#Taking data 135 onwards for testung the model 
X_Test = scale(X_Actual[135:])
Y_Test = tf.one_hot(Y_Actual[135:], 3)


# In[104]:


#Definig place holder to hold data
X = tf.placeholder(tf.float32, shape = (None,4))
Y = tf.placeholder(tf.float32, shape = (None,3))


# In[105]:


W = tf.Variable(tf.random_normal([4,3], stddev = 0.35), name = "weight")
b = tf.Variable(tf.random_normal([3]), name = "bais")


# In[106]:


#Caluclating cross entropy
Y_Prediction = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
cross_entropy = -(Y * tf.log(Y_Prediction))

loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
#tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_Prediction)


# In[141]:


learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# In[142]:


no_of_epochs = 5000
display_step = 1000

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(no_of_epochs):
    sess.run(optimizer , feed_dict = {X : X_Train, Y : Y_Train.eval()})
    if (epoch + 1) % display_step == 0:
        c = sess.run(loss, feed_dict = {X: X_Train, Y: Y_Train.eval()})
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c))

print("***********************************Training Finished!******************************************")
training_loss = sess.run(loss, feed_dict = {X: X_Train, Y: Y_Train.eval()})
print("Training loss=", training_loss, "W=", sess.run(W), "b=", sess.run(b), '\n')


# In[143]:


#Model Validation and Testing
test_loss = sess.run(loss, feed_dict = {X: X_Test, Y: Y_Test.eval()})
print("Test loss=", test_loss, '\n')

correct_prediction = tf.equal(tf.argmax(Y_Prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy(%) =", (sess.run(accuracy, feed_dict={X: X_Test, Y: Y_Test.eval()})) * 100)

prediction = Y_Prediction
print("Predictions =", prediction.eval(feed_dict={X: X_Test}, session=sess))  

