from model import Model
from batcher import Batcher
from Dataset import Data_i2b2


data_i2b2 = Data_i2b2()
# data_i2b2 = data_i2b2.load_data()
# data_i2b2 = data_i2b2.save()
data_i2b2 = data_i2b2.load()
data_i2b2.generate_vocabulary()

train_batcher = Batcher(data_i2b2.train_data, data_i2b2.vocabulary_enumerated)
train_batcher.equalize_classes()

test_batcher = Batcher(data_i2b2.test_data, data_i2b2.vocabulary_enumerated)
test_batcher.equalize_classes()

model = Model(train_batcher, test_batcher)
model.train()
# model.test()
pass
