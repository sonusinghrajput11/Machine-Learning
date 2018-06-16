import collections
import math
import os
import random
import zipfile

from six.moves import urllib
from six.moves import xrange

import numpy as np
import tensorflow as tf

DOWNLOADED_FILENAME = 'SampleText.zip'
def download_file(url_path, expected_bytes):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path, DOWNLOADED_FILENAME)
        
    statinfo = os.stat(DOWNLOADED_FILENAME)
    if statinfo.st_size == expected_bytes:
        print('File downloaded and verified successfully form path:', url_path)
        print('Downloaded File:', DOWNLOADED_FILENAME)
    else:
        print('Failed to verify size:', st_size)           

		def read_words():
    with zipfile.ZipFile(DOWNLOADED_FILENAME) as f:
        firstfile = f.namelist()[0]
        filestring = tf.compat.as_str(f.read(firstfile))
        words = filestring.split()
        
    return words

URL_PATH = 'http://mattmahoney.net/dc/text8.zip'
FILESIZE = 31344016
download_file(URL_PATH, FILESIZE)

vocabulary = read_words()

def build_dataset(words, n_words):
    word_counts = [['UNKNOWN', -1]]
    
    counter = collections.Counter(words)
    word_counts.extend(counter.most_common(n_words - 1))
    
    dictionary = dict()
    
    for word, _ in word_counts:
        dictionary[word] = len(dictionary)
        
    word_indexes = list()
    unknown_count = 0
    index = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            inxed = 0
            unknown_count += 1
        
        word_indexes.append(index)
        
    word_counts[0][1] = unknown_count
    
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return word_counts, word_indexes, dictionary, reverse_dictionary

VOCABULARY_SIZE = 5000
word_counts, word_indexes, dictionary, reverse_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)

del vocabulary

#Global index into words maintedn across batches
global_index = 0

def generate_batch(word_indexes, batch_size, num_skips, skip_window):
    global global_index
    
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    
    span = 2 * skip_window + 1
    
    buffer = collections.deque(maxlen=span)
    
    for _ in range(span):
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)
        
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
                
            target_to_avoid.append(target)
            
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j , 0] = buffer[target]
        
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)
        
    global_index = (global_index + len(word_indexes) - span) % len(word_indexes)
    
    return batch, labels


# In[67]:


batch, labels = generate_batch(word_indexes, 10, 2, 2)

for i in range(9):
    print(reverse_dictionary[batch[i]], ": ", reverse_dictionary[labels[i][0]])

valid_size = 16
valid_window = 100

valid_examples = np.random.choice(valid_window, valid_size, replace = False)

batch_size = 128
embedding_size = 50
skip_window = 2
num_skips = 2

tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

valid_datasets = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embedding_size], -1.0, 1.0))

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))

biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

train_one_hot = tf.one_hot(train_labels, VOCABULARY_SIZE)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hidden_out, labels = train_one_hot))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

l2_norms = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))

normalized_embeddings = embeddings / l2_norms

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_datasets)

similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

num_steps = 20001
init = tf.global_variables_initializer()

with tf.Session() as session:
    init.run()
    
    average_loss = 0
    
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(word_indexes, batch_size, num_skips, skip_window) 
        
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
        
        _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
        average_loss += loss_val
        
        
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
            
        if step % 10000 == 0:
            sim = similarity.eval()
            
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                
                nearest = (-sim[i, :]).argsort()[1 : top_k + 1]
                log_str = 'Nearest to %s' % valid_word
                
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
            print('\n')       

