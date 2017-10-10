import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

sess = tf.Session()

# Load MNIST dataset
data_dir = 'temp'
mnist = read_data_sets(data_dir)
train_x_data = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_x_data = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels

# Parameters
batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_x_data[0].shape[0]
image_height = train_x_data[0].shape[0]
target_size = max(train_labels) + 1
num_channels = 1
generations = 500
evaluate_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size = 100

# Create input / target placeholders
x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(shape=x_input_shape, dtype=tf.float32)
y_target = tf.placeholder(shape=(batch_size), dtype=tf.int32)

evaluate_input_shape = (evaluation_size, image_width, image_height, num_channels)
evaluate_input = tf.placeholder(shape=evaluate_input_shape, dtype=tf.float32)
evaluate_target = tf.placeholder(shape=(evaluation_size), dtype=tf.int32)

# Declare convolution weights and biases with the previously defined parameters
# Recall that a filter is specified with [filter_height, filter_width, in_chans, out_chans]
conv1_weights = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_biases = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_biases = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

# Declare fully-connected weights and biases for the last two layers of the model
resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)

full_input_size = resulting_width * resulting_height * conv2_features
fc1_weights = tf.Variable(tf.truncated_normal([full_input_size, fully_connected_size], stddev=0.1, dtype=tf.float32))
fc1_biases = tf.Variable(tf.truncated_normal([fully_connected_size], stddev=0.1, dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([fully_connected_size, target_size], stddev=0.1, dtype=tf.float32))
fc2_biases = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def model(input_data):
    # Activation maps are specified with shape:
    #
    #       [batch, in_height, in_width, in_chans]
    #
    # Filters are specified with shape:
    #
    #       [filter_height, filter_width, in_chans, out_chans]
    #
    # For max pooling, `ksize` represents the size of the sliding window along each
    # dimension of the input

    # First conv -> relu -> max pool layer
    conv1 = tf.nn.conv2d(input_data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

    # Second conv -> relu -> max pool layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    # Transform output into a 1xn layer for next fully-connected
    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    # First fully-connected layer
    fc = tf.nn.relu(tf.add(tf.matmul(flat_output, fc1_weights), fc1_biases))

    # Second fully-connected layer
    final_model_output = tf.add(tf.matmul(fc, fc2_weights), fc2_biases)

    return final_model_output

model_output = model(x_input)
test_model_output = model(evaluate_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Set up accuracy functions
pred = tf.nn.softmax(model_output)
test_pred = tf.nn.softmax(test_model_output)

def get_accuracy(logits, targets):
    batch_preds = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_preds, targets))

    return 100.0 * num_correct / batch_preds.shape[0]

# Set up optimizer
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = opt.minimize(loss)

sess.run(tf.global_variables_initializer())

train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    rand_index = np.random.choice(len(train_x_data), size=batch_size)
    rand_x = train_x_data[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict=train_dict)

    temp_train_loss, temp_train_preds = sess.run([loss, pred], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    if (i + 1) % evaluate_every == 0:
        eval_index = np.random.choice(len(test_x_data), size=evaluation_size)
        eval_x = test_x_data[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {evaluate_input: eval_x, evaluate_target: eval_y}

        test_preds = sess.run(test_pred, feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)

        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]

        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

evaluation_indices = range(0, generations, evaluate_every)

# Plot the loss over time
plt.plot(evaluation_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot the train and test accuracies
plt.plot(evaluation_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(evaluation_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
