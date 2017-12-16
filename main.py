from Dataset import Data_i2b2
from nltk import word_tokenize, pos_tag

data_i2b2 = Data_i2b2()

tokens = word_tokenize(data_i2b2.train_data["502"]["orig"])
pos_tagged = pos_tag(tokens)

pass