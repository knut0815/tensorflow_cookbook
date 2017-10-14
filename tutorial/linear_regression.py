import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a y-value which is approximately linear but with some random noise
tr_x = np.linspace(-1, 1, 101)
tr_y = 2 * tr_x + np.random.randn(*tr_x.shape) * 0.33

# Scatter plot of x/y pairs
plt.scatter(tr_x, tr_y)
plt.title('Linear Regression Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Create symbolic variables
x = tf.placeholder("float")
y = tf.placeholder("float")

def model(x, w):
    return tf.multiply(x, w)

w = tf.Variable(0.0, name="weights")
y_model = model(x, w)

 # Squared error for cost function
learning_rate = 0.05
cost = tf.reduce_sum(tf.pow(y_model - y, 2.0)) / (2.0 * tr_x.shape[0])
opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = opt.minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:

    # Initialize the variable w
    tf.global_variables_initializer().run()

    for i in range(1000):

        for (rand_x, rand_y) in zip(tr_x, tr_y):
            sess.run(train_step, feed_dict={x: rand_x, y: rand_y})

        if (i + 1) % 50 == 0:
            current_loss = sess.run(cost, feed_dict={x: tr_x, y: tr_y})

            print("Current loss: {}".format(current_loss))

    # It should be something around 2.0
    print(sess.run(w))
