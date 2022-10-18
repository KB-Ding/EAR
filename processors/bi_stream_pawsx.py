# coding=utf-8
import logging
import os

from transformers import DataProcessor
from .utils import InputExample, InputBiStreamExample

logger = logging.getLogger(__name__)


class BiStreamPawsxProcessor(DataProcessor):
    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train'):
        """See base class."""
        examples = []
        if split == 'train':
             for lg in language.split(','):
                lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, lg)))

                for (i, line) in enumerate(lines):
                    guid = "%s-%s-%s" % (split, lg, i)
                    text_a = line[0]
                    text_b = line[1]
                    label1 = str(line[2].strip())
                    text_c = line[3]
                    text_d = line[4]
                    label2 = str(line[5].strip())

                    assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label1, str) and (
                    text_c, str) and isinstance(text_d, str) and isinstance(label2, str)
                    examples.append(
                        InputBiStreamExample(guid=guid, text_a=text_a, text_b=text_b, label1=label1, text_c=text_c,
                                             text_d=text_d, label2=label2, language=lg))
        else:
            for lg in language.split(','):
                lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, lg)))

                for (i, line) in enumerate(lines):
                    guid = "%s-%s-%s" % (split, lg, i)
                    text_a = line[0]
                    text_b = line[1]
                    if split == 'test' and len(line) != 3:
                        label = "0"
                    else:
                        label = str(line[2].strip())
                    assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        return examples

    def get_train_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='train')

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='test')

    def get_translate_train_examples(self, data_dir, language='en'):
        raise NotImplementedError

    def get_translate_test_examples(self, data_dir, language='en'):
        raise NotImplementedError

    def get_pseudo_test_examples(self, data_dir, language='en'):
        raise NotImplementedError

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

