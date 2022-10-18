import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from run.load_data import load_and_cache_examples

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def compute_metrics(preds, labels):
    scores = {
        "acc": (preds == labels).mean(),
        "num": len(
            preds),
        "correct": (preds == labels).sum()
    }
    return scores


def evaluates(args, model, tokenizer, split='train', language='en', lang2id=None, prefix="", output_file=None,
             label_list=None, output_only_prediction=True):
    """Evalute the model."""
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, split=split, language=language,
                                               lang2id=lang2id, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} {} *****".format(prefix, language))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        sentences = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "token_type_ids": batch[2]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                sentences = inputs["input_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        if output_file:
            logger.info("***** Save prediction ******")
            with open(output_file, 'w') as fout:
                pad_token_id = tokenizer.pad_token_id
                sentences = sentences.astype(int).tolist()
                sentences = [[w for w in s if w != pad_token_id] for s in sentences]
                sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
                # fout.write('Prediction\tLabel\tSentences\n')
                for p, l, s in zip(list(preds), list(out_label_ids), sentences):
                    s = ' '.join(s)
                    if label_list:
                        p = label_list[p]
                        l = label_list[l]
                    if output_only_prediction:
                        fout.write(str(p) + '\n')
                    else:
                        fout.write('{}\t{}\t{}\n'.format(p, l, s))

    return results
