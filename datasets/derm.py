from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import os
import random
import json

# debug parameters
TEST_MODE = False

# constants
DS_PATH = "/home/gpu1/datasets/dermnet"

def get_prompt(conv_mode):

    if conv_mode=="default":
        prompt_start = "<image>\nHuman: Classify this image into one of the following categories relating to skin diseases:"
        prompt_end = ". Only output a single final classification label and NOTHING ELSE.\n\n Output:"
    
    elif conv_mode=="instructblip":
        prompt_start = "<image>Question: Can you classify this image into one of the following categories relating to skin diseases:"
        prompt_end = "? Short answer:"

    elif conv_mode=="llava":
        prompt_start = "<image>\nHuman: Classify this image into one of the following categories relating to skin diseases:"
        prompt_end = ". Only output a single final classification label and NOTHING ELSE.\n\n Output:"

    return prompt_start, prompt_end

class CustomDataset(Dataset):

    def __init__(self, processor, image_processor=None, model_config=None, split='train', val_pct=0.1, **kwargs):
        super().__init__()
        self.ds_name = str(Path(os.path.abspath(__file__))).split(os.sep)[-1].split('.')[0]
        self.ds_path = DS_PATH+os.sep+split
        self.file_dir = str(Path(os.path.abspath(__file__)).parent) + os.sep
        self.processor = processor
        self.image_processor = image_processor
        self.model_config = model_config

        # labels
        ds_path_train = Path(DS_PATH + os.sep + 'train')
        ds_path_test = Path(DS_PATH + os.sep + 'test')
        train_labels = [str(item).split(os.sep)[-1] for item in 
                    ds_path_train.iterdir() if item.is_dir()]
        test_labels = [str(item).split(os.sep)[-1] for item in 
                    ds_path_test.iterdir() if item.is_dir()]
        self.labels = list(set(train_labels) & set(test_labels))
        print(self.labels)

        if split == 'train':
            if os.path.exists(self.file_dir+self.ds_name+'.json'):
                self.img_paths = json.load(open(self.file_dir+self.ds_name+'.json', 'r'))['img_paths']
            else:
                # image paths
                self.img_paths = []
                for label in self.labels:
                    self.img_paths += [str(path) for path in Path(
                        self.ds_path+os.sep+label+os.sep).rglob('*.jpg')]
                json_dict = {"img_paths": self.img_paths}    
                with open(self.file_dir+self.ds_name+".json", 'w') as file:
                    json.dump(json_dict, file, indent=4)

            if not os.path.exists(self.file_dir+self.ds_name+'_train.json') or \
                not os.path.exists(self.file_dir+self.ds_name+'_val.json'):
                # image paths
                img_paths = []
                img_paths_train = []
                img_paths_val = []
                for label in self.labels:
                    img_paths += [str(path) for path in Path(
                        self.ds_path+os.sep+label+os.sep).rglob('*.jpg')]
                for path in img_paths:
                    if random.uniform(0, 1) >= val_pct:
                        img_paths_train.append(path)
                    else:
                        img_paths_val.append(path)
                json_dict_train = {"img_paths": img_paths_train}
                json_dict_val = {"img_paths": img_paths_val}    
                with open(self.file_dir+self.ds_name+"_train.json", 'w') as file:
                    json.dump(json_dict_train, file, indent=4)
                with open(self.file_dir+self.ds_name+"_val.json", 'w') as file:
                    json.dump(json_dict_val, file, indent=4)
        elif split == 'test':
            if os.path.exists(self.file_dir+self.ds_name+'_test.json'):
                self.img_paths = json.load(open(self.file_dir+self.ds_name+'_test.json', 'r'))['img_paths']
            else:
                # image paths
                self.img_paths = []
                for label in self.labels:
                    self.img_paths += [str(path) for path in Path(
                        self.ds_path+os.sep+label+os.sep).rglob('*.jpg')]
                json_dict_test = {"img_paths": self.img_paths}    
                with open(self.file_dir+self.ds_name+"_test.json", 'w') as file:
                    json.dump(json_dict_test, file, indent=4)

        if TEST_MODE:
            self.img_paths = [item for item in self.img_paths[:256]]

        self.conv_mode = kwargs.get("conv_mode", "default")
    
    def __getitem__(self, index):        
        inputs = {}

        # image
        img = Image.open(str(self.img_paths[index])).convert('RGB')
        
        # prompt
        labels = [item for item in self.labels]
        random.shuffle(labels)
        prompt_start, prompt_end = get_prompt(self.conv_mode)
        prompt = prompt_start
        for label in labels[:-1]:
            prompt += "\\" + f"{label}" + "\\" + ", "
        prompt += "" + "\\" + f"{labels[-1]}" + "\\"
        prompt += prompt_end
        
        # processing
        inputs = self.processor(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # label
        label = str(self.img_paths[index]).split(os.sep)[-2]

        return inputs, label

    def __len__(self):
        return len(self.img_paths)

    def __str__(self):
        return self.ds_name
    
    def get_train_ds(self):
        return CustomDatasetTrain(self.processor)
    
    def get_val_ds(self):
        return CustomDatasetVal(self.processor)
    
class CustomDatasetTrain(CustomDataset):

    def __init__(self, processor):
        super().__init__(processor=processor)
        self.img_paths = json.load(open(self.file_dir+self.ds_name+'_train.json', 'r'))['img_paths']

class CustomDatasetVal(CustomDataset):

    def __init__(self, processor):
        super().__init__(processor=processor)
        self.img_paths = json.load(open(self.file_dir+self.ds_name+'_val.json', 'r'))['img_paths']