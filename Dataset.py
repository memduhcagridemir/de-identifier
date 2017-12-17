import os
import re
import pickle
import xml.etree.ElementTree as ET
from nltk import word_tokenize, pos_tag

i2b2_DATASET_DIR = "data/i2b2"


class Token(object):
    def __init__(self, raw):
        self.raw = raw
        self.pos = None
        self.text = raw
        self.type = None
        self.length = len(raw)

        annotation = re.search(r"<PHI TYPE=\"([^\"]+?)\">(.+?)</PHI>", raw)
        if annotation:
            self.type = annotation.group(1)
            self.text = annotation.group(2)
            self.length = len(self.text)


class Data_i2b2(object):
    def __init__(self):
        self.train_data = {}
        self.test_data = {}

    def load_data(self):
        for input_file, output_variable in [('deid_surrogate_train_all_version2.xml', self.train_data),
                                            ('deid_surrogate_test_all_groundtruth_version2.xml', self.test_data)]:
            tree = ET.parse(os.path.join(i2b2_DATASET_DIR, input_file))
            records = tree.getroot()

            for record in records:
                record_id = record.get('ID')

                text_element = record[0]

                raw_text = ET.tostring(text_element, encoding='unicode', method='xml')
                raw_text = re.sub(r"<TEXT>\s*((?:.|\s)*)\s*</TEXT>", r"\1", raw_text, flags=re.MULTILINE)

                phi_tag_regex = r'<PHI TYPE=\"[^\"]+?\">(.+?)</PHI>'
                original_text = re.sub(phi_tag_regex, r"\1", raw_text, flags=re.MULTILINE)

                splitted_tokens = []
                phi_tag = re.search(phi_tag_regex, raw_text, flags=re.MULTILINE)
                while phi_tag:
                    splitted_tokens.append(Token(raw_text[:phi_tag.start()]))
                    splitted_tokens.append(Token(raw_text[phi_tag.start():phi_tag.end()]))

                    raw_text = raw_text[phi_tag.end():]
                    phi_tag = re.search(phi_tag_regex, raw_text, flags=re.MULTILINE)

                tokens = []
                for token in splitted_tokens:
                    nltk_tokens = word_tokenize(token.text)
                    for word in nltk_tokens:
                        t = Token(word)
                        t.type = token.type
                        tokens.append(t)

                raw_tokens = [token.text for token in tokens]

                pos_tagged = pos_tag(raw_tokens)
                for i in range(len(tokens)):
                    tokens[i].pos = pos_tagged[i][-1]

                self.train_data[record_id] = {"raw": raw_text, "orig": original_text, "tokens": tokens}

        return self

    def save(self):
        with open(os.path.join(i2b2_DATASET_DIR, 'i2b2.pickle'), 'wb') as pd:
            pickle.dump(self, pd)

        return self

    def load(self):
        with open(os.path.join(i2b2_DATASET_DIR, 'i2b2.pickle'), 'rb') as pr:
            tmp_obj = pickle.load(pr)

        self.__dict__.update(tmp_obj.__dict__)

        return self