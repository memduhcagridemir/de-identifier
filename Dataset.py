import os
import re
import xml.etree.ElementTree as ET

i2b2_DATASET_DIR = "data/i2b2"


class Data_i2b2(object):
    train_data = {}
    test_data = {}

    def __init__(self):
        for input_file, output_variable in [('deid_surrogate_train_all_version2.xml', self.train_data),
                                            ('deid_surrogate_test_all_groundtruth_version2.xml', self.test_data)]:
            tree = ET.parse(os.path.join(i2b2_DATASET_DIR, input_file))
            records = tree.getroot()

            for record in records:
                record_id = record.get('ID')

                text_element = record[0]

                raw_text = ET.tostring(text_element, encoding='unicode', method='xml')
                raw_text = re.sub(r"<TEXT>\s*((?:.|\s)*)\s*</TEXT>", r"\1", raw_text, flags=re.MULTILINE)
                original_text = re.sub(r'<PHI TYPE=\"[^\"]+?\">(.+?)</PHI>', r"\1", raw_text, flags=re.MULTILINE)

                self.train_data[record_id] = {"raw": raw_text, "orig": original_text}
