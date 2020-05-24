import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pdb
import glob
from environments.subproc_physics_env import SubprocPhysicsEnv
from utils.data_loading_utils import load_json_dict, get_time_from_str, scale_position, process_projection
from utils.data_io import read_from_json


class NSDataset(data.Dataset):
    def __init__(self, root_dir, train_val_rate, train=True):
        self.root_dir = root_dir
        self.train_val_rate = train_val_rate
        self.train = train
        self.data = self.load_dataset()

    def load_dataset(self):
        files = glob.glob(os.path.join(self.root_dir, '*.json'))
        train_split = int(len(files) * self.train_val_rate)
        assert len(files) > 0, 'dataset load failed'
        files = files[:train_split] if self.train else files[train_split:]
        all_d = []
        for one_file in files:
            one_d = read_from_json(full_path=one_file, verbose=False)
            all_d += one_d
        return all_d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_d = self.data[idx]
        input_d, output_d = cur_d['input'], cur_d['output']
        input_dict = {
            'force': input_d['force'],
            'contact_points': input_d['list_of_contact_points'],
            'initial_state': input_d['initial_state'],
            'object_num': input_d['object_num'],
        }

        labels = {
            'state': output_d['state'],
            'succ': output_d['succ'],
            'loc': output_d['loc'],
        }

        return input_dict, labels
