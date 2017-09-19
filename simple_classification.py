import tensorflow as tf
import numpy as np

sess = tf.Session()

# Create data from two different normal distributions - the first cluster is
# class 0, the second cluster is class 1
x_vals = np.concatenate((np.random.normal(-1.0, 1.0, 50), np.random.normal(3.0, 1.0, 50)))
y_vals = np.concatenate((np.repeat(0.0, 50), np.repeat(1.0, 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Initialize `A` to ~10 (theoretically, it should converge towards -1)
A = tf.Variable(tf.random_normal(mean=10.0, shape=[1]))

output = tf.add(x_data, A)

# The loss function expects batches of data that have an extra dimension
# associated with them (the batch #) - for now, append an extra dimension
# to the output and target that is always 0
output_expanded = tf.expand_dims(output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

init = tf.initialize_all_variables()
sess.run(init)

# Cross-entropy with unscaled logits
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_expanded, labels=y_target_expanded)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = opt.minimize(cross_entropy)

for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Print the current value of the parameter `A` as well as the current loss
    if (i + 1) % 200 == 0:
        print("Step #{} A = {}".format(i + 1, sess.run(A)))
        print("Loss = {}".format(sess.run(cross_entropy, feed_dict={x_data: rand_x, y_target: rand_y})))
