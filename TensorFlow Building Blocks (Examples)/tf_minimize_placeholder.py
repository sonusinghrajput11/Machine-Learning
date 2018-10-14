import tensorflow as tf
import numpy as np

class TensorFlow:
    def minimize(self):
    
        X_coef = np.array([[1], [-10], [25]])
        X = tf.placeholder(tf.float32, [3,1])
        w = tf.Variable(np.random.rand(1), dtype = tf.float32)
        
        loss = X[0][0] * (w ** 2) + (X[1][0] * w) + X[2][0] 
        
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        
        with tf.Session() as sess :
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for i in range(100):
                sess.run(train, feed_dict={X : X_coef})
                print("Interation {itr} w {res}".format(itr = i, res = w.eval()))
				
def main():
    tensorflow = TensorFlow()
    tensorflow.minimize()

if __name__ == '__main__':
    main()