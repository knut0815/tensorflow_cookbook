import tensorflow as tf
import numpy as np

sess = tf.Session()

seed = 13
tf.set_random_seed(seed)
np.random.seed(seed)

data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(shape=[data_size], dtype=tf.float32)

# First, create a 1D convolution layer
def conv_layer_1d(input_1d, conv_filter):
    # Transform the 1D input into 4D
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    # Perform convolution
    conv_output = tf.nn.conv2d(input_4d, filter=conv_filter, strides=[1, 1, 1, 1], padding='VALID')

    # Now, drop the extra dimensions that we added previously
    conv_output_1d = tf.squeeze(conv_output)

    return conv_output_1d

# Activation maps are specified with shape:
#
#       [batch, in_height, in_width, in_chans]
#
# Filters are specified with shape:
#
#       [filter_height, filter_width, in_chans, out_chans]
#
conv_filter = tf.Variable(tf.random_normal(shape=[1, 5, 1, 1]))
conv_output = conv_layer_1d(x_input_1d, conv_filter)

# TensorFlow's activation functions will act element-wise by default
def relu_layer_1d(input_1d):
    return tf.nn.relu(input_1d)

relu_output = relu_layer_1d(conv_output)

# Now, create a max pooling layer
def max_pooling_layer_1d(input_1d, width):
    # Transform the 1D input into 4D
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    # Perform max pooling - here, `ksize` represents the size of the
    # sliding window along each dimension of the input
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')

    # Now, drop the extra dimensions that we added previously
    pool_output_1d = tf.squeeze(pool_output)

    return pool_output_1d

mp_output = max_pooling_layer_1d(relu_output, width=5)

# The final layer will be a fully-connected layer
def fully_connected_layer_1d(input_layer, num_outputs):
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    print('Input shape:', tf.shape(input_layer))
    print('Weight shape:', weight_shape)
    weights = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])

    # Make input 2D
    input_layer_2d = tf.expand_dims(input_layer, 0)

    # Perform fully-connected operations
    fc_output = tf.add(tf.matmul(input_layer_2d, weights), bias)

    # Drop extra dimensions
    fc_output_1d = tf.squeeze(fc_output)

    return fc_output_1d

fc_output = fully_connected_layer_1d(mp_output, 5)

sess.run(tf.global_variables_initializer())
feed_dict = { x_input_1d: data_1d }

print('Input array of length = 25')
print('Convolution with filter, length = 5, stride = 1, results in an array of length = 21:')
print(sess.run(conv_output, feed_dict=feed_dict))

print('\nInput array of length = 21')
print('ReLU element-wise returns an array of length 21:')
print(sess.run(relu_output, feed_dict=feed_dict))

print('\nInput array of length = 21')
print('Max pooling, length = 5, stride = 1, results in an array of length = 17:')
print(sess.run(mp_output, feed_dict=feed_dict))

print('\nInput array of length = 17')
print('Fully-connected layer on all 4 rows with 5 outputs:')
print(sess.run(fc_output, feed_dict=feed_dict))
