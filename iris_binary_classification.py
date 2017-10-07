import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

sess = tf.Session()

# Load the iris dataset and transform the target data to
# be either 0 or 1, depending on whether the target is a
# setosa or not
iris = datasets.load_iris()
binary_target = np.array([1.0 if x == 0 else 0.0 for x in iris.target])

# Use only two features: petal length and petal width
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Define a linear model of the form: x2 = x1 * A + b
# If we want to find points above or below that line, we see
# whether they are above or below zero when plugged into the
# equation: x2 - x1 * A - b
mult = tf.matmul(x2_data, A)
add = tf.add(mult, b)
output = tf.subtract(x1_data, add)

# Add our sigmoid cross-entropy loss
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y_target)

# Declare an optimization method
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = opt.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})

    # Print the current value of the parameters `A` and `b`
    if (i + 1) % 200 == 0:
        print("Step #{} A = {}, b = {}".format(i + 1, sess.run(A), sess.run(b)))

# Extract the model variables and plot the line on a graph
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, 50)
line_values = [slope * i + intercept for i in x]

setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1.0]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1.0]
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0.0]
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0.0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='non-setosa')
plt.plot(x, line_values, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()
