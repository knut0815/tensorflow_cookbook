import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(x, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden):
    x = tf.nn.dropout(x, p_keep_input)

    h1 = tf.nn.relu(tf.matmul(x, w_h1))
    h1 = tf.nn.dropout(h1, p_keep_hidden)

    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels
iterations = 50
hidden_layer_size = 625
learning_rate = 0.001
decay = 0.9
model_path = "/tmp/mnist_model.ckpt"

# Dimensions
x_col_size = train_x.shape[1] # Should be 784 pixels
y_col_size = train_y.shape[1] # Should be 10 classes

x = tf.placeholder("float", [None, x_col_size])
y = tf.placeholder("float", [None, y_col_size])

w_h1 = init_weights([x_col_size, hidden_layer_size])
w_h2 = init_weights([hidden_layer_size, hidden_layer_size])
w_o = init_weights([hidden_layer_size, y_col_size])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(x, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y))
opt = tf.train.RMSPropOptimizer(learning_rate, decay)
train_step = opt.minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Save the model after training
saver = tf.train.Saver()

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x) + 1, 128)):
                sess.run(train_step, feed_dict={x: train_x[start:end], y: train_y[start:end], p_keep_input: 0.8, p_keep_hidden: 0.5})

            predictions = sess.run(predict_op, feed_dict={x: test_x, p_keep_input: 1.0, p_keep_hidden: 1.0})
            print(i, np.mean(np.argmax(test_y, axis=1) == predictions))

        save_path = saver.save(sess, model_path)
        print("Model saved in file: {}".format(save_path))

def test():
    with tf.Session() as sess:
        print("Loading saved model from file: {}".format(model_path))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)

        print("Running MNIST test set through model...")
        predictions = sess.run(predict_op, feed_dict={x: test_x, p_keep_input: 1.0, p_keep_hidden: 1.0})
        print("Accuracy:", np.mean(np.argmax(test_y, axis=1) == predictions))

if len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    else:
        print('Unknown execution mode: use either train or test')
else:
    train()
