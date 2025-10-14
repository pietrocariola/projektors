from ...datasets.derm import CustomDataset
from ...models.instructblip import CustomModel
from ...models.mlp_cls import PrePostProjCls
from transformers import (
    InstructBlipProcessor,
    BitsAndBytesConfig
)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import NLLLoss
import json
import os
import time
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

# constants
MODEL_PATH = 'Salesforce/instructblip-vicuna-7b'
VAL_PCT = 0.10
BATCH_SIZE = 128
LR = 2e-3
EPOCHS = 1000
PATIENCE = 10
AUX_MODEL_PROJ = 'mlp'
N_SAMPLES = 10
    
# local directory
file_dir = str(Path(__file__).parent)+os.sep

# create directory
os.makedirs(file_dir+"pre", exist_ok=True)
os.makedirs(file_dir+"pre"+os.sep+"train", exist_ok=True)
os.makedirs(file_dir+"pre"+os.sep+"test", exist_ok=True)

os.makedirs(file_dir+"post", exist_ok=True)
os.makedirs(file_dir+"post"+os.sep+"train", exist_ok=True)
os.makedirs(file_dir+"post"+os.sep+"test", exist_ok=True)

# device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# main model initialization
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = InstructBlipProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = CustomModel.from_pretrained(MODEL_PATH, quantization_config=quantization_config)
model.to(device)

# initiliaze train dataset/dataloader
dataset = CustomDataset(processor, split='train', val_pct=VAL_PCT, conv_mode='instructblip')

# dataset split
train_ds = dataset.get_train_ds()
val_ds = dataset.get_val_ds()
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, BATCH_SIZE)

labels_list = [item for item in list(dataset.labels)]

for i in range(N_SAMPLES):

    print(f"\n### SAMPLE {i+1}/{N_SAMPLES} ###\n")

    patience_pre = 0
    patience_post = 0

    # PRE ------------------------------------
    # load json if exists
    json_path_pre = file_dir+f"pre/train/train{i}.json"
    if os.path.exists(json_path_pre):
        metrics_pre = json.load(open(json_path_pre, "r"))
        patience_pre = metrics_pre['patience']
        best_loss_pre = np.min(metrics_pre['val_loss'])
        labels_list = metrics_pre['labels_list']
    else:
        metrics_pre = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        metrics_pre['val_pct'] = VAL_PCT
        metrics_pre['patience_limit'] = PATIENCE
        metrics_pre['batch_size'] = BATCH_SIZE
        metrics_pre['lr'] = LR
        metrics_pre['max_mem0'] = 0
        metrics_pre['time'] = 0
        metrics_pre['labels_list'] = labels_list

    # auxiliary model size
    input_size_pre = model.vision_model.config.hidden_size
    output_size_pre = len(labels_list)

    # save path
    last_cls_path_pre = file_dir+f"pre/train/train{i}_last.pth"
    best_cls_path_pre = file_dir+f"pre/train/train{i}_best.pth"

    # POST ------------------------------------
    # load json if exists
    json_path_post = file_dir+f"post/train/train{i}.json"
    if os.path.exists(json_path_post):
        metrics_post = json.load(open(json_path_post, "r"))
        patience_post = metrics_post['patience']
        best_loss_post = np.min(metrics_post['val_loss'])
        labels_list = metrics_post['labels_list']
    else:
        metrics_post = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        metrics_post['val_pct'] = VAL_PCT
        metrics_post['patience_limit'] = PATIENCE
        metrics_post['batch_size'] = BATCH_SIZE
        metrics_post['lr'] = LR
        metrics_post['max_mem0'] = 0
        metrics_post['time'] = 0
        metrics_post['labels_list'] = labels_list

    # auxiliary model size
    input_size_post = model.config.qformer_config.hidden_size
    output_size_post = len(labels_list)

    # save path
    last_cls_path_post = file_dir+f"post/train/train{i}_last.pth"
    best_cls_path_post = file_dir+f"post/train/train{i}_best.pth"

    # BOTH -----------------------------------------
    # reset memory peak
    torch.cuda.reset_peak_memory_stats()

    # PRE -------------------------------------
    # reload aux model
    cls_pre = PrePostProjCls(input_size_pre, output_size_pre)
    if os.path.exists(last_cls_path_pre):
        checkpoint = torch.load(last_cls_path_pre, map_location=device)
        cls_pre.load_state_dict(checkpoint['model_state_dict'])
    cls_pre.to(device)

    # optimizer
    optim_pre = AdamW(cls_pre.parameters(), lr=metrics_pre['lr'])
    if os.path.exists(last_cls_path_pre):
        optim_pre.load_state_dict(checkpoint['optimizer_state_dict'])

    # epoch
    if os.path.exists(last_cls_path_pre):
        start_epoch_pre = checkpoint['epoch'] + 1
    else:
        start_epoch_pre = 0

    # POST -------------------------------------
    # reload aux model
    cls_post = PrePostProjCls(input_size_post, output_size_post)
    if os.path.exists(last_cls_path_post):
        checkpoint = torch.load(last_cls_path_post, map_location=device)
        cls_post.load_state_dict(checkpoint['model_state_dict'])
    cls_post.to(device)

    # optimizer
    optim_post = AdamW(cls_post.parameters(), lr=metrics_post['lr'])
    if os.path.exists(last_cls_path_post):
        optim_post.load_state_dict(checkpoint['optimizer_state_dict'])

    # epoch
    if os.path.exists(last_cls_path_post):
        start_epoch_post = checkpoint['epoch'] + 1
    else:
        start_epoch_post = 0

    # BOTH -----------------------------------------
    # loss
    loss_fn = NLLLoss()

    model.eval()
    t0_pre = time.time()
    t0_post = time.time()
    max_epoch = np.max([start_epoch_pre, start_epoch_post])
    # training loop
    for epoch in range(max_epoch, EPOCHS):

        print(f"\n### EPOCH {epoch+1}/{EPOCHS} ###\n")

        if patience_pre >= PATIENCE and patience_post >= PATIENCE:
            break

        flag_pat_pre = patience_pre >= PATIENCE
        flag_pat_post = patience_post >= PATIENCE
        
        # train
        
        if not flag_pat_pre:
            train_loss_pre = torch.Tensor([0]).to(device)
            train_hits_pre = torch.Tensor([0]).to(device)
            cls_pre.train()

        if not flag_pat_post:
            train_loss_post = torch.Tensor([0]).to(device)
            train_hits_post = torch.Tensor([0]).to(device)
            cls_post.train()

        for (inputs, labels) in tqdm(train_dl):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                x_pre, x_post = model(**inputs, aux_model_proj=AUX_MODEL_PROJ)
            if not flag_pat_pre:
                x_pre = x_pre.to(dtype=torch.float)
                y_pre = []
                for label in labels: y_pre.append(labels_list.index(label))
                y_pre = torch.tensor(y_pre).to(device)
                y_hat_pre = cls_pre(x_pre)
                loss_pre = loss_fn(y_hat_pre, y_pre)
                optim_pre.zero_grad()
                loss_pre.backward()
                optim_pre.step()
                train_loss_pre += loss_pre
                train_hits_pre += (y_hat_pre.argmax(dim=-1) == y_pre).type(torch.float).sum()
            if not flag_pat_post:
                x_post = x_post.to(dtype=torch.float)
                y_post = []
                for label in labels: y_post.append(labels_list.index(label))
                y_post = torch.tensor(y_post).to(device)
                y_hat_post = cls_post(x_post)
                loss_post = loss_fn(y_hat_post, y_post)
                optim_post.zero_grad()
                loss_post.backward()
                optim_post.step()
                train_loss_post += loss_post
                train_hits_post += (y_hat_post.argmax(dim=-1) == y_post).type(torch.float).sum()
            
        # val
        with torch.no_grad():
            if not flag_pat_pre:
                val_loss_pre = torch.Tensor([0]).to(device)
                val_hits_pre = torch.Tensor([0]).to(device)
                cls_pre.eval()

            if not flag_pat_post:
                val_loss_post = torch.Tensor([0]).to(device)
                val_hits_post = torch.Tensor([0]).to(device)
                cls_post.eval()

            for (inputs, labels) in tqdm(val_dl):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                x_pre, x_post = model(**inputs, aux_model_proj=AUX_MODEL_PROJ)
                if not flag_pat_pre:
                    x_pre = x_pre.to(dtype=torch.float)
                    y_pre = []
                    for label in labels: y_pre.append(labels_list.index(label))
                    y_pre = torch.tensor(y_pre).to(device)
                    y_hat_pre = cls_pre(x_pre)
                    loss_pre = loss_fn(y_hat_pre, y_pre)
                    val_loss_pre += loss_pre
                    val_hits_pre += (y_hat_pre.argmax(dim=-1) == y_pre).type(torch.float).sum()
                if not flag_pat_post:
                    x_post = x_post.to(dtype=torch.float)
                    y_post = []
                    for label in labels: y_post.append(labels_list.index(label))
                    y_post = torch.tensor(y_post).to(device)
                    y_hat_post = cls_post(x_post)
                    loss_post = loss_fn(y_hat_post, y_post)
                    val_loss_post += loss_post
                    val_hits_post += (y_hat_post.argmax(dim=-1) == y_post).type(torch.float).sum()

        # updating training metrics
        # cuda -> cpu
        if not flag_pat_pre:
            train_loss_pre = train_loss_pre.item()
            train_hits_pre = train_hits_pre.item()
            val_loss_pre = val_loss_pre.item()
            val_hits_pre = val_hits_pre.item()

            # metrics - loss
            metrics_pre["train_loss"].append(train_loss_pre/len(train_dl.dataset)*BATCH_SIZE)
            metrics_pre["val_loss"].append(val_loss_pre/len(val_dl.dataset)*BATCH_SIZE)

            # metrics - accuracy
            metrics_pre["train_accuracy"].append(train_hits_pre/len(train_dl.dataset))
            metrics_pre["val_accuracy"].append(val_hits_pre/len(val_dl.dataset))

            if epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cls_pre.state_dict(),
                    'optimizer_state_dict': optim_pre.state_dict(),
                }
                torch.save(checkpoint, best_cls_path_pre)
                best_loss_pre = metrics_pre['val_loss'][-1]
                patience_pre = 0

            if metrics_pre['val_loss'][-1] < best_loss_pre:
                best_loss_pre = metrics_pre['val_loss'][-1]
                patience_pre = 0
                if os.path.exists(best_cls_path_pre):
                    os.remove(best_cls_path_pre)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cls_pre.state_dict(),
                    'optimizer_state_dict': optim_pre.state_dict(),
                }
                torch.save(checkpoint, best_cls_path_pre)
            else:
                patience_pre += 1

            t1_pre = time.time()
            max_mem0 = torch.cuda.max_memory_allocated(device=0)
            metrics_pre['time'] = metrics_pre['time'] + t1_pre - t0_pre
            metrics_pre['max_mem0'] = max(metrics_pre['max_mem0'], max_mem0)
            metrics_pre['patience'] = patience_pre
            metrics_pre['epoch'] = epoch

            print(f"loss: {metrics_pre['val_loss'][-1]} | best: {best_loss_pre}")

            # Use metrics['epoch'] vs checkpoint['epoch'] to compare saves
            with open(json_path_pre, 'w') as file:
                json.dump(metrics_pre, file, indent=4) 

            # save last model   
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': cls_pre.state_dict(),
                'optimizer_state_dict': optim_pre.state_dict(),
            }
            if os.path.exists(last_cls_path_pre):
                os.remove(last_cls_path_pre)
            torch.save(checkpoint, last_cls_path_pre)

        if not flag_pat_post:
            train_loss_post = train_loss_post.item()
            train_hits_post = train_hits_post.item()
            val_loss_post = val_loss_post.item()
            val_hits_post = val_hits_post.item()

            # metrics - loss
            metrics_post["train_loss"].append(train_loss_post/len(train_dl.dataset)*BATCH_SIZE)
            metrics_post["val_loss"].append(val_loss_post/len(val_dl.dataset)*BATCH_SIZE)

            # metrics - accuracy
            metrics_post["train_accuracy"].append(train_hits_post/len(train_dl.dataset))
            metrics_post["val_accuracy"].append(val_hits_post/len(val_dl.dataset))

            if epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cls_post.state_dict(),
                    'optimizer_state_dict': optim_post.state_dict(),
                }
                torch.save(checkpoint, best_cls_path_post)
                best_loss_post = metrics_post['val_loss'][-1]
                patience_post = 0

            if metrics_post['val_loss'][-1] < best_loss_post:
                best_loss_post = metrics_post['val_loss'][-1]
                patience_post = 0
                if os.path.exists(best_cls_path_post):
                    os.remove(best_cls_path_post)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cls_post.state_dict(),
                    'optimizer_state_dict': optim_post.state_dict(),
                }
                torch.save(checkpoint, best_cls_path_post)
            else:
                patience_post += 1

            t1_post = time.time()
            max_mem0 = torch.cuda.max_memory_allocated(device=0)
            metrics_post['time'] = metrics_post['time'] + t1_post - t0_post
            metrics_post['max_mem0'] = max(metrics_post['max_mem0'], max_mem0)
            metrics_post['patience'] = patience_post
            metrics_post['epoch'] = epoch

            print(f"loss: {metrics_post['val_loss'][-1]} | best: {best_loss_post}")

            # Use metrics['epoch'] vs checkpoint['epoch'] to compare saves
            with open(json_path_post, 'w') as file:
                json.dump(metrics_post, file, indent=4) 

            # save last model   
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': cls_post.state_dict(),
                'optimizer_state_dict': optim_post.state_dict(),
            }
            if os.path.exists(last_cls_path_post):
                os.remove(last_cls_path_post)
            torch.save(checkpoint, last_cls_path_post)

print('\nExiting...')