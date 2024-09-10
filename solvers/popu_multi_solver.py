import os
import copy
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics

from tqdm.auto import tqdm

import evaluate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW

from utils import get_num_classes, initialize_networks, get_dataloader, get_tokenizer, load_model, save_model

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)
 
from torch.cuda.amp import GradScaler 

class PopuMultiSolver(object):

    def __init__(self, args, dataset):
        """ Initialize configurations. """
        self.args = args
        self.dataset = dataset
        self.num_class = get_num_classes(args.dataset)
        self.tokenizer = get_tokenizer(args.model, max_length=args.max_seq_len)

        # Load training networks
        self.model = initialize_networks(alg=args.alg, dataset=args.dataset, model=args.model, n_moe=args.n_moe)
        if args.teacher_model is not None:
            self.teacher_model = load_model(args.modeldir, args.teacher_model)
            self.model.load_state_dict(self.teacher_model.state_dict(), strict=False)
        print(self.model)

        # Optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.device = torch.device('cuda')

        self.accuracy = None
        if self.args.dataset == 'imdb':
            self.accuracy = evaluate.load("glue", 'sst2')
        else:
            self.accuracy = evaluate.load("glue", self.args.dataset)
    
    def evaluate(self, model, dataloader):
        
        avg_acc = 0.
        nstep = 0
                
        results = {}
        preds = None
        out_label_ids = None
        pos_count = torch.zeros((model.config.num_hidden_layers)).to(self.device)
        moe_count = torch.zeros((model.config.num_hidden_layers, self.args.n_moe)).to(self.device)

        model.eval()
        with torch.no_grad():
            for it, batch in enumerate(tqdm(dataloader)):
                inputs = {
                        "input_ids": batch['input_ids'].to(self.device),
                        "attention_mask": batch['attention_mask'].to(self.device),
                }
                if 'token_type_ids' in batch:
                    inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                
                outputs = model(**inputs)

                all_pos_masks = self.encoder.all_pos_masks
                for i, mask in enumerate(all_pos_masks):
                    # valid_mask = all_pos_masks
                    valid_mask = ((mask > 0) * inputs['attention_mask']).sum(dim=-1) / inputs['attention_mask'].sum(dim=-1)
                    pos_count[i] += valid_mask.sum()

                all_selected_masks = self.encoder.all_selected_masks
                all_selected_masks = torch.stack(all_selected_masks, dim=1) # (batch, layer, nmoe)
                all_selected_masks = nn.functional.one_hot(all_selected_masks.argmax(dim=-1), num_classes=all_selected_masks.size(-1)).float()

                moe_count += all_selected_masks.sum(dim=0)

                nstep += len(batch['input_ids'])
                if preds is None:
                    preds = outputs.logits.detach().cpu().numpy()
                    out_label_ids = batch["label"].numpy()
                else:
                    preds = np.append(preds, outputs.logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch["label"].numpy(), axis=0)

        print('Routing results # Position')
        routings = pos_count / nstep
        speedup = routings.mean().item()
        routings = routings.detach().cpu().numpy().tolist()
        routings = [ '%.2f' % elem for elem in routings ]
        print('base pos\t', routings, speedup)

        print('Routing results # MoE')
        routings = moe_count / nstep
        print('base moe\t', routings.mean(dim=0))

        if self.args.dataset in ['sst2', 'mrpc', 'imdb', 'mnli', 'qnli', 'stsb']: # classification
            if self.args.dataset == 'stsb':
                preds = preds[:, 0]
            else:
                preds = np.argmax(preds, axis=1)
            results = self.accuracy.compute(predictions=preds, references=out_label_ids)
        task2measure = {'sst2': 'accuracy', 'mrpc': 'f1', 'imdb': 'accuracy', 'mnli': 'accuracy', 'qnli': 'accuracy', 'stsb': 'pearson'}
        return task2measure[self.args.dataset], results[task2measure[self.args.dataset]]
                
                

    def run(self):
        """ Start federated learning scenario """
        # Load global validation set
        train_loader, test_loader = get_dataloader(dataset=self.dataset, train_bs=self.args.batch_size, test_bs=self.args.batch_size)
        t_total = len(train_loader) * self.args.epochs

        # optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "mask_" not in n],
                "lr": self.args.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "mask_" in n],
                "lr": 0.1,
            },
        ]
         
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8) # get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        if 'roberta' in self.args.model:
            self.encoder = self.model.roberta.encoder
        elif 'bert' in self.args.model:
            self.encoder = self.model.bert.encoder

        cur_step = 0
        best_acc = -1
        self.model = self.model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device).eval()
        for epoch in range(self.args.epochs):
            writer = {'loss': 0., 'acc': 0., 'step': 0}
            self.model.train()
            for it, batch in enumerate(tqdm(train_loader)):
                
                inputs = {
                    "input_ids": batch['input_ids'].to(self.device),
                    "attention_mask": batch['attention_mask'].to(self.device),
                    "labels": batch['label'].to(self.device),
                }
                if 'token_type_ids' in batch:
                    inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                outputs = self.model(**inputs)
                loss = outputs.loss

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)

                all_pos_masks = self.encoder.all_pos_masks
                pos_reg = 0.
                all_weights = 0 
                for i, mask in enumerate(all_pos_masks):
                    valid_mask = (mask * inputs['attention_mask']).sum(dim=-1) / inputs['attention_mask'].sum(dim=-1)
                    pos_reg += ((valid_mask - self.args.p_decay) ** 2).mean() * (i+1)
                    all_weights += (i+1)
                pos_reg = pos_reg / all_weights
                
                moe_reg = 0.
                all_selected_masks = self.encoder.all_selected_masks
                all_selected_masks = torch.stack(all_selected_masks, dim=1) # (batch, layer, nmoe)
                selected_probs = F.softmax(all_selected_masks, dim=-1) # (batch, layer, nmoe)
                selected_masks = nn.functional.one_hot(selected_probs.argmax(dim=-1), num_classes=all_selected_masks.size(-1)).float()

                # kd_loss = 0.
                temperature = 1.0
                kd_loss = F.kl_div(F.log_softmax(outputs.logits, dim=-1), F.softmax(teacher_outputs.logits, dim=-1).detach(), reduction='batchmean')

                moe_reg = (selected_probs.mean(dim=0) - (1. / all_selected_masks.size(-1))) ** 2
                selected_probs = selected_probs.mean(dim=0)
                moe_reg = (selected_masks.mean(dim=0) * selected_probs).sum(dim=-1).mean()

                if epoch == (self.args.epochs - 1):
                    loss = loss + kd_loss * (temperature ** 2)
                else:
                    loss = loss + pos_reg * self.args.reg_weight + moe_reg + kd_loss * (temperature ** 2)

                # Model Updates
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                writer['loss']  += loss.mean().item()
                writer['step']  += 1
                cur_step += 1
                
                if (it+1) % self.args.logging_step == 0:
                    metric, test_acc = self.evaluate(self.model, test_loader)
                    print(f'Epoch ({epoch}, {it+1} step) test {metric} {test_acc}')
                    if test_acc > best_acc:
                        best_acc = test_acc
                        save_model(self.model, self.args.modeldir, f'{self.args.dataset}_{self.args.model.split("/")[-1]}_{self.args.alg}_{test_acc}.pt')
                    self.model.train()
                    
            metric, test_acc = self.evaluate(self.model, test_loader)
            avg_loss = writer['loss'] / writer['step']
            print(f'Epoch ({epoch}) avg loss {avg_loss} test {metric} {test_acc}')
            if test_acc > best_acc:
                best_acc = test_acc
                save_model(self.model, self.args.modeldir, f'{self.args.dataset}_{self.args.model.split("/")[-1]}_{self.args.alg}_{test_acc}.pt')

