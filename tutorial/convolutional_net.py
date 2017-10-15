
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(x, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    '''
    Convolutional layer #1
    '''
    l1a = tf.nn.relu(tf.nn.conv2d(x, w1,                      # l1a shape=(?, 28, 28, 32)
                     strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                     strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    '''
    Convolutional layer #2
    '''
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                     strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                     strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    '''
    Convolutional layer #3
    '''
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                     strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                     strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    '''
    Fully-connected layer
    '''
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)

    return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print("Train x shape:", train_x.shape)
print("Test x shape:", test_x.shape)
print("Resizing inputs...")
train_x = train_x.reshape(-1, 28, 28, 1)  # 28x28x1 input img
test_x = test_x.reshape(-1, 28, 28, 1)    # 28x28x1 input img
print("New train x shape:", train_x.shape)
print("New test x shape:", test_x.shape)

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, 10])

# Input tensors will have shape:  [batch, in_height, in_width, in_channels]
# Filters will have shape:        [filter_height, filter_width, in_channels, out_channels]
#
w1 = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 layers in activation map
w2 = init_weights([3, 3, 32, 64])      # 3x3x32 conv, 64 layers in activation map
w3 = init_weights([3, 3, 64, 128])     # 3x3x32 conv, 128 layers in activation map
w4 = init_weights([128 * 4 * 4, 625])  # fc 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])          # fc 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(x, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y))
opt = tf.train.RMSPropOptimizer(0.001, 0.9)
train_step = opt.minimize(cost)
pred = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x)+1, batch_size))

        for start, end in training_batch:
            sess.run(train_step, feed_dict={x: train_x[start:end],
                                            y: train_y[start:end],
                                            p_keep_conv: 0.8,
                                            p_keep_hidden: 0.5})

        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(test_y[test_indices], axis=1) ==
                         sess.run(pred, feed_dict={x: test_x[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))
