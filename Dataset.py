import os
import xml.etree.ElementTree as ET

i2b2_DATASET_DIR = "data/i2b2"


class i2b2_Dataset(object):
    train_data = {}
    test_data = {}

    def __init__(self):
        for input_file, output_variable in [('deid_surrogate_test_all_groundtruth_version2.xml', self.test_data),
                                            ('deid_surrogate_train_all_version2.xml', self.train_data)]:
            tree = ET.parse(os.path.join(i2b2_DATASET_DIR, input_file))
            records = tree.getroot()

            for record in records:
                record_id = record.get('ID')

                text_element = record[0]
                element_list = ET.tostringlist(text_element, encoding='utf8', method='xml')

                # ignore some parts:
                # ["<?xml version='1.0' encoding='utf8'?>\n", '<TEXT', '>', '\n', '<PHI', ' TYPE="ID"', '>', ..
                # , '/95\nCC :\n[ report_end ]\n', '</TEXT>', '\n']
                output_variable[record_id] = "".join(element_list[4:-2])


i2b2_data = i2b2_Dataset()
print(len(i2b2_data.train_data))
print()
print(len(i2b2_data.test_data))