import os
import math
import json

import logging
from tqdm import tqdm 

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler 

import wandb

logger = logging.getLogger(__name__)
wandb.init(project="mamba-runs")

class TrainerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, trainset, evalset, train_config):
        self.model = model
        self.trainset = trainset
        self.evalset = evalset
        self.config = train_config
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.model.to(self.device)

        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        if not os.path.exists(self.config.generated_text_path):
            os.makedirs(self.config.generated_text_path)

    def save_checkpoints(self, ckpt_id):
        model = self.model
        ckpt_folder = self.config.ckpt_path
        torch.save(model.state_dict(), f"{ckpt_folder}/{ckpt_id}")
    
    def generate_text(self, model, num_tokens):
        idx = torch.zeros((1,1), dtype=torch.long).to(self.device)
        token_ids = model.generate(idx, num_tokens)
        text = "".join([self.trainset.decoder[c.item()] for c in token_ids.squeeze()])
        return text

    
    def train(self):
        config = self.config
        model = self.model
        optimizer = model.mamba_optimizer(config)

        lr_steps = int(len(self.trainset) / config.train_batch_size * config.max_epoch)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, lr_steps)
        #lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=steps_per_epoch, epochs=config.max_epoch)
        
        def train_loop(train_dataloader, epoch_idx=1):
            model.train()
            
            for itr, (x,y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train'):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                _, loss = model(x, y)

                loss.backward()
                optimizer.step()
                scheduler.step()
                train_metrics = {"train/train_loss": loss, "train/train_lr": scheduler.get_last_lr()}
                wandb.log(train_metrics)

                if itr%1000 == 0:
                    generated_text = self.generate_text(model, num_tokens=config.num_generated_tokens)
                    state_generated_text = {"epoch":epoch_idx,
                                            "generated_text": generated_text,
                                            "train_itr": itr}
                    with open(f'{config.generated_text_path}/mamba_generated_text.json', "a") as json_file:
                        state_generated_text_dict = json.dumps(state_generated_text)
                        json_file.write(state_generated_text_dict + '\n')



        def eval_loop(eval_dataloader):
            model.eval()
            losses = []
            for _, (x, y) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='Eval'):
                x = x.to(self.device)
                y = y.to(self.device)

                _, loss = model(x, y)
                losses.append(loss.item())
                val_metrics = {"val/val_loss": loss}
                wandb.log(val_metrics)                
            return float(np.mean(losses))
        

        train_dataloader = DataLoader(
            self.trainset,
            batch_size = config.train_batch_size,
            num_workers = config.num_workers,
            drop_last = True,
        )

        eval_dataloader = DataLoader(
            self.evalset,
            batch_size =  config.eval_batch_size,
            num_workers = config.num_workers,
            drop_last= True
        )

        best_loss = float('inf')
        for epoch in range(config.max_epoch):
            logger.info(f"===============Epoch:{epoch+1}/{config.max_epoch}=============")
            
            train_loop(train_dataloader, epoch_idx=(epoch+1))
            eval_loss = eval_loop(eval_dataloader)

            goodModel = eval_loss < best_loss
            if config.ckpt_path is not None and goodModel:
                wandb.run.summary["best_val_loss"] = eval_loss
                best_loss = eval_loss
                self.save_checkpoints(f"_{config.max_epoch}epoch_best_model")

        self.save_checkpoints(f"_{config.max_epoch}epoch_last_model")
        wandb.finish()
