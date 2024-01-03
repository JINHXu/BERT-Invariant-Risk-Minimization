from dataset_utils import create_datasets# , BasicDataset
import settings
from utils import calc_steps_params # , general_cm_to_fig, cm_to_fig, to_tuple
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW, SGD
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup, \
    AutoModelForSequenceClassification, AutoTokenizer
# from attribution_utils import *
# from lit_utils import *
from train_utils import IRMTrainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import re
import torch
import json
from torch.utils.tensorboard import SummaryWriter
# from lit_nlp import server_flags


def run_irm(out_dir='.', dataset='HatEval', num_labels=2, pretrained_model='bert-base-cased', seed=None,
            # Training params
            bs_train=32, bs_val=32, train_size=None, val_size=None,
            eval_every_x_epoch=0.2, epochs=4, warm_up_epochs=2, early_stopping=3,
            reg=1e3, warm_up_reg=1.0, gradient_checkpoint=False,
            # optimizer params
            optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
            epsilon=1e-8, weight_decay=0.01, amsgrad=False,
            lr_scheduling=False, lr_scheduling_rate=0.1
            ):
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()
    checkpoint_dir = os.path.sep.join([out_dir, "checkpoints"])

    # file paths
    ethos_train_path = '/content/ethos_train.txt'
    ethos_val_path = '/content/ethos_val.txt'
    davidson_train_path = '/content/davidson_train.txt'
    davidson_val_path = '/content/davidson_val.txt'
    val_ood_path = '/content/val_ood.txt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                          gradient_checkpointing=gradient_checkpoint).to(device=device)

    rng = np.random.RandomState(seed)

    ethos_train = create_datasets(ethos_train_path)
    davidson_train = create_datasets(davidson_train_path)
    
    ethos_val = create_datasets(ethos_val_path)
    davidson_val = create_datasets(davidson_val_path)
    
    val_ood = create_datasets(val_ood_path)
    
    ds_train = ethos_train + davidson_train
    ds_val = ethos_val+davidson_val
    ds_val_ood = val_ood
    
    dl_train = [DataLoader(env, batch_size=bs_train, shuffle=True) for env in ds_train]
    dl_val = [DataLoader(env, batch_size=bs_val) for env in ds_val]
    dl_val_ood = [DataLoader(env, batch_size=bs_val) for env in ds_val_ood]

    batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train, eval_every_x_epoch, warm_up_epochs, epochs)

    if optimizer_type.lower() == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay,
                          amsgrad=amsgrad)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise AttributeError('only SGD and Adam supported for now')

    if batches_per_step is not None:
        total_steps = (warm_up_steps + steps) * batches_per_step
    else:
        total_steps = (warm_up_steps + steps) * len(dl_train[0])

    if lr_scheduling:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(lr_scheduling_rate * total_steps),
                                                    num_training_steps=total_steps)
    else:
        scheduler = None

    # trainer = IRMTrainer(model, num_labels, tokenizer, optimizer, scheduler, device=device)

    trainer = IRMTrainer(model, num_labels, tokenizer, weights=None, optimizer=optimizer, scheduler=scheduler,
                         device=device)

    res = trainer.fit(dl_train, dl_val, dl_val_ood, warm_up_steps=warm_up_steps, irm_steps=steps,
                      warm_up_reg=warm_up_reg, irm_reg=reg, checkpoint_dir=checkpoint_dir,
                      early_stopping=early_stopping,
                      max_batches=batches_per_step)

    # save last checkpointed model
    checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
    saved_state = torch.load(checkpoint_filename)
    model.load_state_dict(saved_state['model_state_dict'])
    save_experiment(out_dir, run_config, res, model)
    writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]))
    for k in ['train_env_prob', 'val_env_prob', 'val_ood_env_prob', 'bias_tokens']:
        run_config[k] = str(run_config[k])
    writer.add_hparams(run_config,
                       {'hparam/train_accuracy': res.train_acc[-1], 'hparam/val_accuracy': res.test_acc[-1],
                        'hparam/val_ood_accuracy': res.test_acc_ood[-1]})

    return res


def test_irm(test_file, test_dir, out_dir='.', seed=None,
             bs_test=32, reg=1e3
             ):
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    rng = np.random.RandomState(seed)

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = pretrained_cfg['pretrained_model'], pretrained_cfg['num_labels']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir).to(device=device)
    rng = np.random.RandomState(seed)

    # create biased datasets by appending unused tokens to hypothesis
    ds_test = create_datasets(test_file)
    dl_test = [DataLoader(env, batch_size=bs_test) for env in ds_test]

    tester = IRMTrainer(model, num_labels, tokenizer, device=device)
    res = tester.test(dl_test, reg=reg)
    loss = res.loss
    acc = res.accuracy
    # cm_fig = cm_to_fig(res.cm)
    pred_prob = res.pred_prob
    test_res = {'loss': loss, 'accuracy': acc, 'predicted probabilities': pred_prob}

    save_experiment(out_dir, run_config, test_res)
    output_filename = f'{os.path.sep.join([out_dir, "confusion_matrix"])}.png'
    # cm_fig.savefig(output_filename)
    return res

def save_experiment(out_dir, config, res, model=None):
    if not isinstance(res, dict):
        res = res._asdict()
    output = dict(
        config=config,
        results=res
    )
    output_filename = f'{os.path.sep.join([out_dir, "run_output"])}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')

    if model is not None:
        model.save_pretrained(out_dir)
        print('*** Model saved')


def parse_cli():
    p = argparse.ArgumentParser(description='Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-irm', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_irm)

    # <editor-fold desc="run-irm params">
    # data and model params
    sp_exp.add_argument('--out-dir', type=str,
                        help='Name dir to save results',
                        default='.')
    sp_exp.add_argument('--num-labels', type=int,
                        help='Number of label types',
                        default=2)
    sp_exp.add_argument('--pretrained-model', type=str,
                        help='Name of the huggingface model', default='bert-base-cased')
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        required=False)
    # training params
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-val', type=int, help='Val batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--train-size', type=int, help='Train data size')
    sp_exp.add_argument('--val-size', type=int, help='Val data size')
    sp_exp.add_argument('--eval-every-x-epoch', type=float, help='Evaluate on validation every x fraction of an epoch',
                        default=0.2)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of constrained epochs', default=4)
    sp_exp.add_argument('--warm-up-epochs', type=int,
                        help='Maximal number of warm up epochs', default=2)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without improvement', default=3)
    sp_exp.add_argument('--reg', type=float,
                        help='IRM regularization weight', default=1e3)
    sp_exp.add_argument('--warm-up-reg', type=float,
                        help='IRM regularization weight for warm up', default=1.0)
    sp_exp.add_argument('--gradient-checkpoint',
                        help='Checkpoint gradient to fit big batches in memory', action='store_true')

    # optimization params
    sp_exp.add_argument('--optimizer-type', '-ot', type=str,
                        help='Which type of optimizer to use', default="Adam")
    sp_exp.add_argument('--lr', '-lr', type=float,
                        help='Learning rate', default=1e-5)
    sp_exp.add_argument('--momentum', '-m', type=float,
                        help='Momentum for SGD', default=0.9)
    sp_exp.add_argument('--beta1', '-b1', type=float,
                        default=0.9)
    sp_exp.add_argument('--beta2', '-b2', type=float,
                        default=0.999)
    sp_exp.add_argument('--epsilon', '-eps', type=float,
                        default=1e-6)
    sp_exp.add_argument('--weight-decay', '-wd', type=float,
                        default=0.01)
    sp_exp.add_argument('--amsgrad', action='store_true')
    sp_exp.add_argument('--lr-scheduling', action='store_true')
    sp_exp.add_argument('--lr-scheduling-rate', type=float,
                        default=0.1)

    sp_test = sp.add_parser('test-irm', help='Evaluate model on test or validation')
    sp_test.set_defaults(subcmd_fn=test_irm)

    # <editor-fold desc="test-irm params">
    sp_test.add_argument('test_file', type=str,
                         help='File to evaluate model on')
    sp_test.add_argument('test_dir', type=str,
                         help='Name dir to load fine-tuned model')

    sp_test.add_argument('--out-dir', type=str,
                         help='Name dir to save results',
                         default='.')

    sp_test.add_argument('--seed', '-s', type=int, help='Random seed',
                         required=False)
    sp_test.add_argument('--bs-test', type=int, help='Batch size',
                         default=32, metavar='BATCH_SIZE')
    sp_test.add_argument('--reg', type=float,
                         help='IRM regularization weight', default=1e3)

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
