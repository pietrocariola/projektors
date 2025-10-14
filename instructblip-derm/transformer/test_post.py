from ...models.instructblip import CustomModel
from ...datasets.derm import CustomDataset
from ...models.transformer_linear_cls import PrePostProjCls2
from transformers import (
    InstructBlipProcessor,
    BitsAndBytesConfig
)
from torch.utils.data import DataLoader
import json
import torch
import time
from tqdm import tqdm
from pathlib import Path
import os

# constants
MODEL_PATH = 'Salesforce/instructblip-vicuna-7b'
BATCH_SIZE = 32
AUX_MODEL_PROJ = 'transformer'
PRE_POST = 'post'

# local directory
file_dir = str(Path(__file__).parent)+os.sep

# scan for models
cls_tested = [str(path).replace('test', 'train').replace('.pth', '_best.pth') \
              for path in Path(file_dir+PRE_POST+'/test/').rglob('test*.pth')]
cls_models = [str(path) for path in Path(file_dir+PRE_POST+'/train/').rglob('train*_best.pth')]
cls_models = list(set(cls_models) - set(cls_tested))

# device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# main model initialization
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = InstructBlipProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = CustomModel.from_pretrained(MODEL_PATH, quantization_config=quantization_config)

# initiliaze dataset/dataloader
eval_ds = CustomDataset(processor, split='test', conv_mode='instructblip')
eval_dl = DataLoader(eval_ds, BATCH_SIZE)

# label list
labels_list = json.load(open(file_dir+PRE_POST+'/train/train0.json', 'r'))['labels_list']

# auxiliary model
input_size = model.config.qformer_config.hidden_size
hidden_size = model.vision_model.config.hidden_size
output_size = len(labels_list)
cls = PrePostProjCls2(input_size, hidden_size, output_size)

# setup model
model.eval()
model.to(device)

for i, cls_path in enumerate(cls_models):

    print(f"### SAMPLE {i+1}/{len(cls_models)} ###")

    sample_num = cls_path.split('train')[-1].split('_')[0]

    # metrics
    metrics = {}

    gts = []
    preds = []

    # reset memory
    torch.cuda.reset_peak_memory_stats()

    # label list
    labels_list = json.load(open(cls_path.replace('_best.pth', '.json'), 'r'))['labels_list']

    # load parameters
    cls.load_state_dict(torch.load(cls_path)['model_state_dict'])

    # setup initialization
    cls.eval()
    cls.to(device)

    t0 = time.time()

    # eval
    with torch.no_grad():
        eval_hits = torch.Tensor([0]).to(device)
        for (inputs, labels) in tqdm(eval_dl):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if PRE_POST.lower() == 'pre':
                x, _ = model(**inputs, aux_model_proj=AUX_MODEL_PROJ)
            elif PRE_POST.lower() == 'post':
                _, x = model(**inputs, aux_model_proj=AUX_MODEL_PROJ)
            x = x.to(dtype=torch.float)
            y = []
            for label in labels: y.append(labels_list.index(label))
            y = torch.tensor(y).to(device)
            y_hat = cls(x)
            eval_hits += (y_hat.argmax(dim=-1) == y).type(torch.float).sum()
            gts += y.tolist()
            preds += y_hat.argmax(dim=-1).tolist()

    t1 = time.time()
    delta_t = t1 - t0

    # cuda -> cpu
    eval_hits = eval_hits.item()
    metrics["eval_accuracy"] = eval_hits/len(eval_dl.dataset)
    metrics["gts"] = gts
    metrics["preds"] = preds
    metrics['time'] = delta_t
    metrics['labels'] = labels_list
    max_mem0 = torch.cuda.max_memory_allocated(device=0)
    metrics['max_mem0'] = max_mem0

    # create directory
    os.makedirs(file_dir+PRE_POST+"/test", exist_ok=True)
    with open(file_dir+PRE_POST+f"/test/test{sample_num}.json", 'w') as f:
        json.dump(metrics, f, indent=4)

print('\nExiting ...')