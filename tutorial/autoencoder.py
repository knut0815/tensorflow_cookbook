import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def vis(images, save_name):
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
    gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    for g, count in zip(gs,range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count,:].reshape((28,28)))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name + '_vis.png')

mnist_width = 28
n_visible = mnist_width * mnist_width   # 784
n_hidden = 50
corruption_level = 0.3
epochs = 100
batch_size = 128

# Create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# Create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')

# Create nodes for hidden variables
W_init_max = 4.0 * np.sqrt(6.0 / (n_visible + n_hidden)) # around ~0.27
print("W_init_max:", W_init_max)
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')
print("W shape:", W.shape)  # (784, 500)
print("b shape:", b.shape)  # (500, )

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X  # corrupted X
    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)        # hidden state: (n, 500)
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input

    return Z

# Build model graph
Z = model(X, mask, W, b, W_prime, b_prime)

# Create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2.0))
opt = tf.train.GradientDescentOptimizer(0.02)
train_step = opt.minimize(cost)
pred = Z

# Load MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(epochs):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_step, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))

    # Save the predictions for 100 images
    mask_np = np.random.binomial(1, 1 - corruption_level, teX[:100].shape)
    predicted_imgs = sess.run(pred, feed_dict={X: teX[:100], mask: mask_np})
    input_imgs = teX[:100]

    # Plot the reconstructed images
    vis(predicted_imgs,'pred')
    vis(input_imgs,'in')

print('Done')
