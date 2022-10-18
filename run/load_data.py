import logging
import os

import torch
from torch.utils.data import TensorDataset

from processors.process import PROCESSORS
from processors.utils import convert_examples_to_features, convert_examples_to_bi_features

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, task, tokenizer, split='train', language='en', lang2id=None, evaluate=False):
    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = PROCESSORS[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    lc = '_lc' if args.do_lower_case else ''
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}{}".format(
            split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(language),
            lc,
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if split == 'train':
            examples = processor.get_train_examples(args.data_dir, language)
        elif split == 'translate-train':
            examples = processor.get_translate_train_examples(args.data_dir, language)
        elif split == 'translate-test':
            examples = processor.get_translate_test_examples(args.data_dir, language)
        elif split == 'dev':
            examples = processor.get_dev_examples(args.data_dir, language)
        elif split == 'pseudo_test':
            examples = processor.get_pseudo_test_examples(args.data_dir, language)
        else:
            examples = processor.get_test_examples(args.data_dir, language)
        if args.bi_stream and split == 'train':
            features = convert_examples_to_bi_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
                pad_on_left=False,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                lang2id=lang2id,
            )
        else:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
                pad_on_left=False,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                lang2id=lang2id,
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for {}.".format(args.task_name))

    if args.bi_stream and split == 'train':
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)

        if output_mode == "classification":
            all_labels1 = torch.tensor([f.label1 for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
                                    all_input_ids1, all_attention_mask1,all_token_type_ids1, all_labels1)
        else:
            raise NotImplementedError
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
