import tensorflow as tf
import numpy as np

class TensorFlow:
    def run(self):
    
        X = tf.Variable(np.random.rand(1,1));
        
        loss = (X*X) - (4*X) + 4 
        
        train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        
        with tf.Session() as sess :
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for i in range(100):
                sess.run(train)
                print("Interation {itr} X {res}".format(itr = i, res = X.eval()))
				

tensorflow = TensorFlow()
tensorflow.run()