import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
from pprint import pprint

# Start a graph session
sess = tf.Session()

# Check if data was downloaded, otherwise download it and save for future use
save_file_name = os.path.join('temp','temp_spam_data.csv')
if not os.path.exists('temp'):
    os.makedirs('temp')

if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')

    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x)>=1]

    # And write to csv
    with open(save_file_name, 'w', newline='') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# Relabel 'spam' as 1, 'ham' as 0
target = [1 if x=='spam' else 0 for x in target]

# Normalize text
# Lower case
texts = [x.lower() for x in texts]

# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# Plot histogram of text lengths
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins=25)
plt.title('Histogram of # of Words in Texts')
#plt.show()

# Setup vocabulary processor
sentence_size = 25
min_word_freq = 3
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words
vocab_processor.transform(texts)
embedding_size = len([x for x in vocab_processor.transform(texts)])
print('Embedding size:', embedding_size)

# Split up data set into train / test
train_indices = np.random.choice(len(texts), round(len(texts) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

# Declare the embedding matrix for words
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Init placeholders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# Map indices of the words in the sentence to the one-hot encoded vectors
# of our identity matrix - we create the sentence vector by summing up the
# aforementioned word vectors
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

# Perform logistic regression
x_col_sums_2d = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2d, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
pred = tf.sigmoid(model_output)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_step = opt.minimize(loss)

sess.run(tf.global_variables_initializer())

losses = []
train_acc_all = []
train_acc_avg = []

for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]
    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})

    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    losses.append(temp_loss)

    if (ix + 1) % 10 == 0:
        print('Training observation #{}: Loss = {}'.format(ix + 1, temp_loss))

    # Keep a trailing average of the past 50 observations accuracy

    [[temp_pred]] = sess.run(pred, feed_dict={x_data: t, y_target: y_data})

    # Get true / false if prediction is accurate
    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))
        
