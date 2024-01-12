import os
import random
import configparser
import argparse
import time

import numpy as np

import torch

from mamba_model import mambaCofig, MambaLM
from trainer import Trainer, TrainerConfig
from prepare_dataset import CharDataset


import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(seed: int):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True


SEED = 1738
seed_everything(SEED)

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("-c","--config",dest="filename",help="Pass config file",metavar="FILE")
args = parser.parse_args()
config.read(args.filename)


    #################################
    #       Model                   #
    #################################

block_size = int(config['model_config']['block_size'])
d_model = int(config['model_config']['d_model'])
d_state = int(config['model_config']['d_state'])
d_conv = int(config['model_config']['d_conv'])
expand = int(config['model_config']['expand'])       
dt_min = float(config['model_config']['dt_min'])
dt_max = float(config['model_config']['dt_max'])
dt_init = config['model_config']['dt_init']
dt_scale = float(config['model_config']['dt_scale'])
conv_bias = config.getboolean('model_config', 'conv_bias')
dt_init_floor = float(config['model_config']['dt_init_floor'])
in_out_proj_bias = config.getboolean('model_config','in_out_proj_bias')
n_layer = int(config['model_config']['n_layer'])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32

    #################################
    #       Dataset                 #
    #################################


dataset_path = config['dataset']['dataset_path']

text = open(dataset_path, 'r').read()
n = len(text)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]
train_dataset = CharDataset(train_data, block_size)
eval_dataset = CharDataset(val_data, block_size)

    #################################
    #       Train Args              #
    #################################
max_epoch = int(config['training_config']['max_epoch'])
train_batch_size = int(config['training_config']['train_batch_size'])
eval_batch_size = int(config['training_config']['eval_batch_size'])
num_workers = int(config['training_config']['num_workers'])
learning_rate = float(config['training_config']['learning_rate'])
weight_decay = float(config['training_config']['weight_decay'])
beta_1 = float(config['training_config']['beta_1'])
beta_2 = float(config['training_config']['beta_2'])
ckpt_path = config['training_config']['ckpt_path']
vocab_size = train_dataset.get_vocab_size() if train_dataset.get_vocab_size() > eval_dataset.get_vocab_size() else eval_dataset.get_vocab_size() 

num_generated_tokens = int(config['generation_config']['num_generated_tokens'])
generated_text_path = config['generation_config']['generated_text_path']
    #################################
    #       Training                #
    #################################

model_config = mambaCofig(
        vocab_size=vocab_size,
        d_model = d_model,
        d_state = d_state,
        d_conv = d_conv,
        expand = expand,
        n_layer= n_layer,
        device = device,
        dtype = dtype,
        dt_min= dt_min,
        dt_max = dt_max,
        dt_init = dt_init,
        dt_scale= dt_scale,
        dt_init_floor= dt_init_floor,
        conv_bias= conv_bias,
        in_out_proj_bias= in_out_proj_bias,
        block_size = block_size
    )

model = MambaLM(model_config)

training_config = TrainerConfig(
        max_epoch = max_epoch,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        num_workers = num_workers,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        AdamwBetas = (beta_1, beta_2),
        num_generated_tokens=num_generated_tokens,
        ckpt_path = ckpt_path,
        generated_text_path = generated_text_path)


if __name__ == "__main__":
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
	
        start.record()
        trainer = Trainer(model, train_dataset, eval_dataset, training_config)
        trainer.train()
        end.record()

        torch.cuda.synchronize()
	
        logger.info(f"Training time:{start.elapsed_time(end)}")

    else:
        start = time.time()

        trainer = Trainer(model, train_dataset, eval_dataset, training_config)
        trainer.train()

        elapsed_time = time.time() - start
        logger.info(f"Training time:{elapsed_time}")