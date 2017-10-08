import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
sess = tf.Session()

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# Create an 80-20 train-test split and normalize the x features
# to be between 0..1 via min-max scaling
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare the batch size and placeholders for the data and target
batch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare our model variables with the appropriate shape
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

# Declare our model in two steps:
# 1) Create the hidden layer output
# 2) Create the final output
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# MSE loss
loss = tf.reduce_mean(tf.square(y_target - final_output))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train_step = opt.minimize(loss)
sess.run(tf.global_variables_initializer())

losses = []
test_losses = []

for i in range(500):
    rand_indices = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_indices]
    rand_y = np.transpose([y_vals_train[rand_indices]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Save the training loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    losses.append(np.sqrt(temp_loss))

    # Run the test set through our model and save the loss
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_losses.append(np.sqrt(test_temp_loss))

    if (i + 1) % 50 == 0:
        print("Generation: {}, Loss = {}".format(i + 1, temp_loss))

plt.plot(losses, 'k--', label='Train Loss')
plt.plot(test_losses, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
