import pickle

from Dataset import Data_i2b2
from model import Model
from batcher import Batcher
from nltk import word_tokenize, pos_tag


def generate_vocabulary(dataset):
    vocabulary = {}
    for record_id, record in dataset.items():
        for i in range(len(record['tokens'])):
            if i < 2 or i >= len(record['tokens']) - 2:
                continue

            if record['tokens'][i].type:
                for window_index in range(i - 2, i + 3):
                    if not record['tokens'][window_index].type:
                        if record['tokens'][window_index].text in vocabulary:
                            vocabulary[record['tokens'][window_index].text] += 1
                        else:
                            vocabulary[record['tokens'][window_index].text] = 1

    del_words = []
    for word, word_count in vocabulary.items():
        if word_count < 5:
            del_words.append(word)

    for word in del_words:
        del vocabulary[word]

    enum_vocabulary = {}
    word_id = 0
    for word, freq in vocabulary.items():
        enum_vocabulary[word] = word_id
        word_id += 1

    return enum_vocabulary

data_i2b2 = Data_i2b2()
# data_i2b2 = data_i2b2.load_data()
# data_i2b2 = data_i2b2.save()
data_i2b2 = data_i2b2.load()


vocab = generate_vocabulary(data_i2b2.train_data)

train_batcher = Batcher(data_i2b2.train_data, vocab)
test_batcher = Batcher(data_i2b2.test_data, vocab)
model = Model(train_batcher, test_batcher)

model.train()
pass