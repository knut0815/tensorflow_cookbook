import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(x, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(x, w_h))

    # Note that we dont take the softmax at the end because our cost fn does that for us
    return tf.matmul(h, w_o)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
iterations = 1000
print_every = 10
batch_size = 128
learning_rate = 0.05
momentum = 0.9
hidden_layer_size = 625
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

# Dimensions
x_col_size = train_x.shape[1] # Should be 784 pixels
y_col_size = train_y.shape[1] # Should be 10 classes

x = tf.placeholder("float", [None, x_col_size])
y = tf.placeholder("float", [None, y_col_size])
w_h = init_weights([x_col_size, hidden_layer_size])
w_o = init_weights([hidden_layer_size, y_col_size])

py_x = model(x, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y))
opt = tf.train.MomentumOptimizer(learning_rate, momentum)
train_step = opt.minimize(cost)
pred = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        if (i + 1)  % print_every == 0:
            print("Accuracy at step #{}: {}".format(i + 1, np.mean(np.argmax(test_y, axis=1) == sess.run(pred, feed_dict={x: test_x}))))
