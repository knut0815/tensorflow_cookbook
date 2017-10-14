import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W):
    return tf.matmul(X, W)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
iterations = 1000
print_every = 100
batch_size = 100
learning_rate = 0.05
momentum = 0.9
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

# Dimensions
x_col_size = train_x.shape[1] # Should be 784 pixels
y_col_size = train_y.shape[1] # Should be 10 classes

X = tf.placeholder("float", [None, x_col_size])
Y = tf.placeholder("float", [None, y_col_size])
W = init_weights([x_col_size, y_col_size])

py_x = model(X, W)

# Compute mean cross entropy (softmax is applied internally)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
opt = tf.train.MomentumOptimizer(learning_rate, momentum)
train_step = opt.minimize(cost)

# At predict time, evaluate the argmax of the logistic regression
pred = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

        if (i + 1)  % print_every == 0:
            print(i, np.mean(np.argmax(test_y, axis=1) == sess.run(pred, feed_dict={X: test_x})))
