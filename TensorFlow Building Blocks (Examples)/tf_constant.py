import tensorflow as tf
import numpy as np

class TensorFlow:
    def run(self):
        num1 = np.array([1,2,3])
        num2 = np.array([1,2,3])
        
        input1 = tf.constant(num1);
        input2 = tf.constant(num2);
        
        result = input1 + input2;
        
        with tf.Session() as sess :
            init = tf.global_variables_initializer()
            sess.run(init)
            return sess.run(result)
			

tensorflow = TensorFlow()
print(tensorflow.run())