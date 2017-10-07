import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()

# The number of data points that will be fed through the graph
batch_size = 20

# Declare the data, placeholders, and variable - a shape of `None`
# means that TensorFlow will figure out the dimensions for us
x_vals = np.random.normal(1.0, 0.0, 100)
y_vals = np.repeat(10.0, 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))

# Here, we need matrix multiplication because we are working with
# batches now
output = tf.matmul(x_data, A)

init = tf.initialize_all_variables()
sess.run(init)

# Our loss function will be the mean of all of the L2 losses across
# the entire batch - we can calculate this using TensorFlow's
# `reduce_mean` function
loss = tf.reduce_mean(tf.square(output - y_target))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = opt.minimize(loss)

# Store the loss calculated for each batch so that we can plot
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 5 == 0:
        print("Step #{} A = {}".format(i + 1, sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print("Loss = {}".format(temp_loss))

        loss_batch.append(temp_loss)

plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss')
plt.show()
