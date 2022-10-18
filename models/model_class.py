from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification
)

from models.bert_ept_apt import Bert_ept_apt
from models.xlmr_ept_apt import Xlmr_ept_apt


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "bert-ept-apt": (BertConfig, Bert_ept_apt, BertTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "xlmr-ept-apt": (XLMRobertaConfig, Xlmr_ept_apt, XLMRobertaTokenizer),

}