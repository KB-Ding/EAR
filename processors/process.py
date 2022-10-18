from processors.bi_stream_pawsx import BiStreamPawsxProcessor
from processors.bi_stream_xnli import BiStreamXnliProcessor
from processors.pawsx import PawsxProcessor
from processors.xnli import XnliProcessor

PROCESSORS = {
    'aug_ori_xnli': BiStreamXnliProcessor,
    'aug_ori_pawsx': BiStreamPawsxProcessor,
    'xnli': XnliProcessor,
    'pawsx': PawsxProcessor,
}
