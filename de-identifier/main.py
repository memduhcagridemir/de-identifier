from Model import Model
from Batcher import Batcher
from Dataset import Data_i2b2

# test datadaki pos/neg orani 1/20

data_i2b2 = Data_i2b2()
# data_i2b2 = data_i2b2.load_data()
# data_i2b2 = data_i2b2.save()
data_i2b2 = data_i2b2.load()
data_i2b2.generate_vocabulary()

num_true = 0
num_false = 0
for key, value in data_i2b2.test_data.items():
    for token in value['tokens']:
        if token.type is None:
            num_false += 1
        else:
            num_true += 1

print(num_true, num_false)
exit()

train_batcher = Batcher(data_i2b2.train_data, data_i2b2.vocabulary_enumerated)
# train_batcher.equalize_classes()

test_batcher = Batcher(data_i2b2.test_data, data_i2b2.vocabulary_enumerated)
# test_batcher.equalize_classes()

model = Model(train_batcher, test_batcher)
model.train()
# model.test()
pass
