import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Run this to ensure TensorFlow 2.x is used
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

# Dummy run

# training_text = ["My name is Nuo", "I am from China", "I love travelling and experience culture"]
# test_text = ["Nuo loves travel and experience", "Nuo loves dog"]

# oov_token is used to place holder out of vocabulary word, which apprears in test set, not in training set.
# if removed, the unrecorginized word in test set will be skipped
# tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
# tokenizer.fit_on_texts(training_text)
# index_word = tokenizer.index_word
# print(index_word)

# training_seq = tokenizer.texts_to_sequences(training_text)
# training_pad = pad_sequences(training_seq, maxlen=6)
# print("padded training seq:\n", training_pad)

# test_seq = tokenizer.texts_to_sequences(test_text)
# print("padded test seq:\n", pad_sequences(test_seq, maxlen=6))
 

# Predict news headline is sarcastic

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
#     -O /tmp/sarcasm.json

with open("/tmp/sarcasm.json", 'r') as f:
  datastore = json.load(f)

sentences = []
labels = []
for data in datastore:
  sentences.append(data['headline'])
  labels.append(data['is_sarcastic'])

# Split training and test data
training_text = sentences[0:training_size]
training_label = labels[0:training_size]
test_text = sentences[training_size:]
test_label = labels[training_size:]

# create tokenizer on the whole dataset, including training and test
tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok);
tokenizer.fit_on_texts(sentences)
# print(tokenizer.word_index)
training_pad = pad_sequences(tokenizer.texts_to_sequences(training_text), maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_pad = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Need this block to get it to work with TensorFlow 2.x
import numpy as np
training_pad = np.array(training_pad)
training_label = np.array(training_label)
test_pad = np.array(test_pad)
test_label = np.array(test_label)


# embedding_dim: used for output dimension
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x=training_pad, y=training_label, verbose=2, epochs=30, validation_data=(test_pad, test_label))

# Save embedding dimension
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

# Plot model evaluation history
import matplotlib.pyplot as plt

def plot_graph(history, string):
  plt.plot(history[string])
  plt.plot(history['val_'+string])
  plt.xlabel('epoch')
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graph(model.history.history, 'loss')
plot_graph(model.history.history, 'accuracy')


# Predict sentence
predict_sentence = ["Cows lose their jobs as milk prices drop", "granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
predict_pad = pad_sequences(tokenizer.texts_to_sequences(predict_sentence), maxlen=max_length, truncating=trunc_type, padding=padding_type)
print(model.predict(predict_pad))

# Compare reversed token text vs raw text
reverse_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])

def reverse_sentence(pad):
  return ' '.join([reverse_word_index.get(i, '?') for i in pad])

print("Reversed text: ", reverse_sentence(training_pad[0]))
print("Raw text: ", training_text[0])

