import tensorflow as tf
import numpy as np

# Start the graph session and create the data, placeholders, and
# variable `A`
sess = tf.Session()
x_vals = np.random.normal(1.0, 0.1, 100)
y_vals = np.repeat(10.0, 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

# Create a simple linear model
output = tf.multiply(x_data, A)

# L2 loss
loss = tf.square(output - y_target)

# Initialize all TensorFlow variables
init = tf.initialize_all_variables()
sess.run(init)

opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = opt.minimize(loss)

# Run the optimizer
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Print the current value of the parameter `A` as well as the current loss
    if (i + 1) % 25 == 0:
        print("Step #{} A = {}".format(i + 1, sess.run(A)))
        print("Loss = {}".format(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
