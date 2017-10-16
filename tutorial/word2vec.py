import collections
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pprint import pprint

batch_size = 20

# Dimension of the embedding vector - 2 is too small to get any meaningful embeddings,
# but let's make it 2 for simple visualization
embedding_size = 2

# Number of negative examples to sample
num_sampled = 15

# Sample sentences
sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]

# Contenate all words across all sentences into one, big list
words = " ".join(sentences).split()

# Convert the `words` list into a list of tuples, which associates each unique word with a count
count = collections.Counter(words).most_common()
print ("Word count", count[:5])

# Build dictionaries
rdic = [i[0] for i in count]
dic = {w: i for i, w in enumerate(rdic)}    # Dictionary that maps: word -> unique ID
vocabulary_size = len(dic)

# Make indexed word data
data = [dic[word] for word in words]
print('Sample data', data[:10], [rdic[t] for t in data[:10]])

# Let's make a training data for window size 1 for simplicity
# ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
cbow_pairs = [];
for i in range(1, len(data)-1) :
    cbow_pairs.append([[data[i-1], data[i+1]], data[i]]);
print('Context pairs', cbow_pairs[:10])

# Let's make skip-gram pairs: (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
skip_gram_pairs = [];
for c in cbow_pairs:
    skip_gram_pairs.append([c[1], c[0][0]])
    skip_gram_pairs.append([c[1], c[0][1]])
print('Skip-gram pairs', skip_gram_pairs[:5])

def generate_batch(size):
    assert size < len(skip_gram_pairs)
    x_data= []
    y_data = []
    r = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)
    for i in r:
        x_data.append(skip_gram_pairs[i][0])    # n dim
        y_data.append([skip_gram_pairs[i][1]])  # n, 1 dim

    return x_data, y_data

# generate_batch test
print ('Batches (x, y)', generate_batch(3))

# Reshape data into [batch_size, 1] for nn.nce_loss
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.device('/cpu:0'):
    # Look up embeddings for inputs
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch
# This does the magic: tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes ...)
# It automatically draws negative samples when we evaluate the loss
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, vocabulary_size))
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(12000):
        batch_inputs, batch_labels = generate_batch(batch_size)
        _, loss_val = sess.run([train_op, loss], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})

        if step % 100 == 0:
          print("Loss at ", step, loss_val)

    # Final embeddings are ready for you to use
    trained_embeddings = embeddings.eval()

# Show word2vec if 2-dimensional
if trained_embeddings.shape[1] == 2:
    labels = rdic
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig("word2vec.png")
