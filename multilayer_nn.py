import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import random
import requests

birth_weight_file = 'birth_weight.csv'

# Download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    with open(birth_weight_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)
        f.close()

# Read birth weight data into memory
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]

# Extract y-target (birth weight)
y_vals = np.array([x[8] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

sess = tf.Session()
batch_size = 100

seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

def init_weights(shape, st_dev):
    weights = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return weights

def init_biases(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return bias

x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

def fully_connected_layer(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.relu(layer)

# Create our model with hidden layer sizes 25, 10, 3
weights_1 = init_weights(shape=[7, 25], st_dev=10.0)
biases_1 = init_biases(shape=[25], st_dev=10.0)
layer_1 = fully_connected_layer(x_data, weights_1, biases_1)

weights_2 = init_weights(shape=[25, 10], st_dev=10.0)
biases_2 = init_biases(shape=[10], st_dev=10.0)
layer_2 = fully_connected_layer(layer_1, weights_2, biases_2)

weights_3 = init_weights(shape=[10, 3], st_dev=10.0)
biases_3 = init_biases(shape=[3], st_dev=10.0)
layer_3 = fully_connected_layer(layer_2, weights_3, biases_3)

# Create the final output layer
weights_4 = init_weights(shape=[3, 1], st_dev=10.0)
biases_4 = init_biases(shape=[1], st_dev=10.0)
final_output = fully_connected_layer(layer_3, weights_4, biases_4)

# Declare L1 loss
loss = tf.reduce_mean(tf.abs(y_target - final_output))
opt = tf.train.AdamOptimizer(learning_rate=0.05)
train_step = opt.minimize(loss)
sess.run(tf.global_variables_initializer())

losses = []
test_losses = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Save train loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    losses.append(temp_loss)

    # Save test loss
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_losses.append(test_temp_loss)

    if (i + 1) % 25 == 0:
        print('Generation: {}, Loss = {}'.format(i + 1, temp_loss))

plt.plot(losses, 'k-', label='Train Loss')
plt.plot(test_losses, 'r--', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
