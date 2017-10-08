import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

batch_size = 50
a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))
x = np.random.normal(2.0, 0.1, 500)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare our two models
sigm_act = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
relu_act = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

# The loss functions will be the average L2 norm between the
# model output and the value 0.75
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigm_act, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_act, 0.75)))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step_sigm = opt.minimize(loss1)
train_step_relu = opt.minimize(loss2)
sess.run(tf.global_variables_initializer())

losses_sigm = []
losses_relu = []
acts_sigm = []
acts_relu = []

for i in range(750):
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])
    sess.run(train_step_sigm, feed_dict={x_data: x_vals})
    sess.run(train_step_relu, feed_dict={x_data: x_vals})

    losses_sigm.append(sess.run(loss1, feed_dict={x_data: x_vals}))
    losses_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))

    acts_sigm.append(np.mean(sess.run(sigm_act, feed_dict={x_data: x_vals})))
    acts_relu.append(np.mean(sess.run(relu_act, feed_dict={x_data: x_vals})))

plt.plot(acts_sigm, 'k-', label='Sigmoid Activation')
plt.plot(acts_relu, 'r--', label='ReLU Activation')
plt.ylim([0.0, 1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show()
plt.plot(losses_sigm, 'k-', label='Sigmoid Loss')
plt.plot(losses_relu, 'r--', label='ReLU Loss')
plt.ylim([0.0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
