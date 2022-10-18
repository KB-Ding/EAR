# coding=utf-8
from transformers import WEIGHTS_NAME

from run.load_data import load_and_cache_examples
from run.evaluate import evaluates
from run.opts import opts
from run.train import train
from models.model_class import MODEL_CLASSES
from processors.process import PROCESSORS
from utils.seeds import set_seed


import argparse
import glob
import logging
import os
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    opts(parser)
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare dataset
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info("config = {}".format(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    '''
    train start
    '''
    if args.do_train:
        if args.init_checkpoint:
            logger.info("loading from folder {}".format(args.init_checkpoint))
            model = model_class.from_pretrained(
                args.init_checkpoint,
                config=config,
                cache_dir=args.init_checkpoint,
            )
        else:
            logger.info("loading from existing model {}".format(args.model_name_or_path))
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        model.to(args.device)
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, split=args.train_split,
                                                language=args.train_language, lang2id=lang2id, evaluate=False)
        global_step, tr_loss, best_checkpoint = train(args, train_dataset, model, tokenizer, lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" best checkpoint = {}".format(best_checkpoint))
    '''
    save start
    '''
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    last_ckp_dir = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(last_ckp_dir) and args.local_rank in [-1, 0]:
            os.makedirs(last_ckp_dir)

        logger.info("Saving model checkpoint to %s", last_ckp_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(last_ckp_dir)
        tokenizer.save_pretrained(last_ckp_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(last_ckp_dir)
        tokenizer = tokenizer_class.from_pretrained(last_ckp_dir)
        model.to(args.device)
    '''
    evaluate start
    '''
    # Evaluation 默认best: (先从init_checkpoint，在best-check，再最后一轮的)，evaluate过程中选目录下所有ckp进行比较, 英文dev集上的结果
    if args.init_checkpoint:
        best_checkpoint = args.init_checkpoint
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-last')


    if args.do_eval and args.local_rank in [-1, 0]:
        best_score = 0.0
        tokenizer = tokenizer_class.from_pretrained(best_checkpoint, do_lower_case=args.do_lower_case)
        checkpoints = [best_checkpoint]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints on the dev set: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            if args.save_dev:
                result = evaluates(args, model, tokenizer, split='dev', language=args.train_language, lang2id=lang2id,
                                  prefix=prefix)
                if result['acc'] > best_score:
                    best_checkpoint = checkpoint
                    best_score = result['acc']

            else:
                total = total_correct = 0.0
                for language in args.predict_languages.split(','):
                    result = evaluates(args, model, tokenizer, split=args.test_split, language=language,
                                       lang2id=lang2id, prefix=prefix)
                    total += result['num']
                    total_correct += result['correct']
                test_score = total_correct / total
                if test_score > best_score:
                    best_checkpoint = checkpoint
                    best_score = test_score

        logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint,best_score))


    # Prediction 触发为真
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(best_checkpoint, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)
        output_predict_file = os.path.join(args.output_dir, args.test_split + '_results.txt')
        total = total_correct = 0.0
        with open(output_predict_file, 'a') as writer:
            writer.write('======= Predict using the model from {} for test:\n'.format(best_checkpoint))
            for language in args.predict_languages.split(','):
                output_file = os.path.join(args.output_dir, 'test-{}.tsv'.format(language))
                result = evaluates(args, model, tokenizer, split=args.test_split, language=language, lang2id=lang2id,
                                  prefix='best_checkpoint', output_file=output_file, label_list=label_list)
                writer.write('{}={}\n'.format(language, result['acc']))
                logger.info('{}={}'.format(language, result['acc']))
                total += result['num']
                total_correct += result['correct']
            writer.write('total={}\n'.format(total_correct / total))


if __name__ == "__main__":
    main()
