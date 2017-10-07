import tensorflow as tf
import numpy as np

sess = tf.Session()
x_vals = np.random.normal(1.0, 0.1, 100)
y_vals = np.repeat(10.0, 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
batch_size = 25
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

# Split into training and test sets
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

output = tf.matmul(x_data, A)
loss = tf.reduce_mean(tf.square(output - y_target))
sess.run(tf.global_variables_initializer())

opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = opt.minimize(loss)

for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Print the current value of the parameter `A` as well as the current loss
    if (i + 1) % 25 == 0:
        print("Step #{} A = {}".format(i + 1, sess.run(A)))
        print("Loss = {}".format(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

# Evaluate our model on both the train and test sets
mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print("MSE test: {}".format(np.round(mse_test, 2)))
print("MSE train: {}".format(np.round(mse_train, 2)))
