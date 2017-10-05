import tensorflow as tf
import sys
sys.path.append('/home/dashmoment/workspace/tf_tools')
from utility import netfactory as nf
from utility import model_zoo as mz

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10])


#Optimizee
net = nf.fc_layer(x, 256, "fc1")
net = nf.fc_layer(net, 10, "fc2",  activat_fn = None)
output = tf.nn.softmax(net)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))


variables_names = [v.name for v in tf.trainable_variables()]
variable = tf.trainable_variables()
print(variables_names)
print(variable)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_xs, batch_ys = mnist.train.next_batch(100)


values = sess.run(variables_names)
g = sess.run(tf.gradients(cross_entropy, variable), feed_dict={x: batch_xs, y_: batch_ys})
print(len(g))

