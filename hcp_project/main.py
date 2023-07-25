import pandas as pd
import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
from tensorflow.keras import layers

# read text data from data_list.txt
my_file = open("data_list.txt", "r")
path_data = my_file.read()
paths = path_data.split("\n")
print(paths)
my_file.close()

contents = []

for path in paths:
    with open(path, 'r', encoding='utf-8') as f:  # encoding='utf-8'不行
        for line in f.readlines():
            contents.append(line)
    f.close()

# read rate scores from rate.txt
rate = []
my_file = open("rate.txt", "r")
rate_data = my_file.read()
rate = rate_data.split("\n")
print(rate)
my_file.close()

# build data frame ("test", "label")
data = pd.DataFrame([])
data["text"] = contents
data["label"] = rate
print(data)

# tokenize the text data
token = re.compile('[A-Za-z]+|[!?,.()]')
def reg_text(text):
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text

data['text'] = data.text.apply(reg_text)
word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)

# make dataset for training, make sure the size of each book's vector is the same
max_word = len(word_set) + 1
word_list = list(word_set)

word_index = dict((word, word_list.index(word) + 1) for word in word_list)
data_ok = data.text.apply(lambda x: [word_index.get(word, 0) for word in x])

maxlen = max(len(x) for x in data_ok)
data_ok = tf.keras.preprocessing.sequence.pad_sequences(data_ok.values, maxlen=maxlen)
print("Shape of data_ok: ", data_ok.shape)


# build ther model and train
def train_model():
    model = keras.Sequential()
    model.add(layers.Embedding(max_word, 16, input_length=maxlen))
    model.add(layers.Bidirectional(layers.LSTM(64,
                         dropout=0.2,
                         recurrent_dropout=0.5)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc'])
    return model

model = train_model()

print(model.summary())

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, min_lr=0.00001)
history = model.fit(data_ok,
                     data.label.values,
                     epochs=50,
                     batch_size=128,
                     validation_split=0.2,
                     callbacks=[learning_rate_reduction])

# save the model as model1.keras
model.save('model/model1.keras')
