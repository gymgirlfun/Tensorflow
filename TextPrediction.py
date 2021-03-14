import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
    -O /tmp/irish-lyrics-eof.txt
    
tokenizer = Tokenizer()

data = open('/tmp/irish-lyrics-eof.txt').read()

corpus = data.lower().split("\n")
print(corpus)

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)    

# Split training data and labels
# Caveat: texts_to_sequences(), param should be a list of words '[line]'. If pass only 'line', the sentence will be tokenized based on letter rather than word
# Example: This is an apple
# Input_sequences = tokenize([[This is], [This is an], [This is an apple]])
input_sequences = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
	  n_gram_sequence = token_list[:i+1]
	  input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
# xs = [[This], [This is], [This is an]]
# labels = [[is], [an], [apple]]
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

# covert into binary matrix
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


# Some ramdom test
print(tokenizer.word_index['come'])
print(tokenizer.word_index['all'])
print(tokenizer.word_index['ye'])
print(tokenizer.word_index['maidens'])
print(tokenizer.word_index['young'])
print(tokenizer.word_index['and'])
print(tokenizer.word_index['fair'])
print(xs[5])
print(ys[5])
print(labels[5])

# train model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=10, verbose=1)
#print model.summary()
print(model)

# Plot summary
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()
plot_graphs(history, 'accuracy')


# Let predict next word
seed_text = "come all ye maidens young and"
next_words = 100
print(tokenizer.word_index)

for _ in range(next_words):  # Try to predict 100 possiblilty
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  predicted = model.predict_classes(token_list, verbose=0)
  print("predicted: ", predicted)
  
  reverse_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])
  output_word = reverse_word_index[predicted[0]] #predicted is a list, so use predicted[0] to be dict key
  seed_text += " " + output_word
print(seed_text)
