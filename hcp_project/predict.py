import pandas as pd
import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# read text data for testing from data_list.txt
my_file = open("data_list_short_test.txt", "r")
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


# build data frame ("test", "label")
data = pd.DataFrame([])
data["text"] = contents
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



model = keras.models.load_model('model/model3.keras')
result = model.predict(data_ok)

print(result)



# read rate scores from rate.txt
# rate = []
# my_file = open("rate_short.txt", "r")
# rate_data = my_file.read()
# rate = rate_data.split("\n")
# print(rate)
# my_file.close()
#
# lb = LabelEncoder()
# rate = lb.fit_transform(rate)
#
# print(result[0][0])
# #
# print(lb.inverse_transform(result[0]))