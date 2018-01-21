import random
import re


class Batcher(object):
    def __init__(self, dataset, vocab):
        self.batch_num = -1
        self.batch_length = 8192

        self.dataset = dataset

        self.vocabulary = vocab

        self.input_tokens = []
        self.output_tokens = []

        for record_id, record in self.dataset.items():
            inputs = []
            outputs = []
            for token in record["tokens"]:
                row = []

                vocab_feature = [0] * len(self.vocabulary)
                if token.text in self.vocabulary:
                    vocab_feature[self.vocabulary[token.text]] = 1
                row += vocab_feature

                pos_feature = [0] * 36
                pos_feature[Batcher.__get_pos_enum(token.pos)] = 1
                row += pos_feature

                type_feature = [0] * 4
                type_feature[Batcher.__get_type_enum(token.text)] = 1
                row += type_feature

                row += [token.length]

                inputs.append(row)

                output = [0] * 2
                output[int(token.type is not None)] = 1
                outputs.append(output)

            for i in range(2, len(inputs) - 2):
                self.input_tokens.append(inputs[i - 2:i + 3])
                self.output_tokens.append(outputs[i])

    def __get_pos_enum(pos):
        try:
            return ['UNK', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP',
                    'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                    'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'].index(pos)
        except ValueError:
            return 0

    def __get_type_enum(text):
        if re.match(r"[0-9]+", text):
            # all digits
            return 1
        if re.match(r"[a-z]+", text, flags=re.IGNORECASE):
            return 2
        if re.match(r"[0-9]+(?:/|-|\.)[0-9]+", text):
            return 3
        return 0

    def get_all(self):
        return self.input_tokens, self.output_tokens

    def get_number_of_batches(self):
        return len(self.input_tokens) // self.batch_length

    def reset_batches(self):
        self.batch_num = -1

    def has_more_batches(self):
        return self.batch_num < self.get_number_of_batches()

    def get_next_batch(self):
        self.batch_num += 1

        if self.batch_num > self.get_number_of_batches():
            return None

        return self.input_tokens[self.batch_num * self.batch_length:(self.batch_num+1) * self.batch_length], \
            self.output_tokens[self.batch_num * self.batch_length:(self.batch_num+1) * self.batch_length]

    def equalize_classes(self):
        yes_tokens = []
        no_tokens = []
        for i in range(len(self.output_tokens)):
            if self.output_tokens[i][0] == 1:
                no_tokens.append(i)
            else:
                yes_tokens.append(i)

        delete_ids = []
        while len(no_tokens) > len(yes_tokens):
            index_to_delete = random.randrange(0, len(no_tokens))
            delete_ids.append(no_tokens[index_to_delete])
            del no_tokens[index_to_delete]

        delete_ids.sort(reverse=True)
        for index_to_delete in delete_ids:
            del self.input_tokens[index_to_delete]
            del self.output_tokens[index_to_delete]

        return self
