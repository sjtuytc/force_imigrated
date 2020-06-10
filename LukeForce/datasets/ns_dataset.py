import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pdb
import random
import glob
from utils.data_io import read_from_json, read_from_pkl
from utils.environment_util import NormEnvState
from utils.tensor_utils import norm_tensor
import numpy as np


class NSDataset(data.Dataset):
    def __init__(self, obj_name, root_dir, train_val_rate, train=True, data_statistics=None):
        self.obj_name = obj_name
        self.root_dir = root_dir
        self.train_val_rate = train_val_rate
        self.train = train
        self.data = self.load_dataset()
        if self.train:
            self.data_statistics = self.cal_statistics()
        else:
            self.data_statistics = data_statistics

    def load_dataset(self):
        files = glob.glob(os.path.join(self.root_dir, '*.pkl'))
        assert len(files) > 0, 'dataset load failed'
        data_file = files[0]
        all_data = read_from_pkl(full_path=data_file)
        if self.obj_name is not None:
            new_data = []
            for one_data in all_data:
                if one_data['object_name'] == self.obj_name:
                    new_data.append(one_data)
            all_data = new_data
        random.seed(0)
        random.shuffle(all_data)
        split_ind = int(self.train_val_rate * len(all_data))
        print("Total number:", len(all_data), "; split number:", split_ind, ".")
        final_d = all_data[:split_ind] if self.train else all_data[split_ind:]
        return final_d

    def cal_statistics(self):
        print("Calculating data statistics.")
        all_forces, all_cps, all_position, all_rotation, all_velocity, all_omega = [], [], [], [], [], []
        for idx, ele in enumerate(self.data):
            forces, cps, position, rotation = ele['force_applied'][0], ele['model_contact_points'][0], ele['model_position'][0], \
                                              ele['model_rotation'][0]
            all_forces.append(forces)
            all_cps.append(cps)
            all_position.append(position)
            all_rotation.append(rotation)
            all_position.append(ele['initial_position'])
            all_rotation.append(ele['initial_rotation'])

        all_forces, all_cps, all_position, all_rotation, all_velocity, all_omega = np.array(all_forces), np.array(all_cps), \
            np.array(all_position), np.array(all_rotation), np.array(all_velocity), np.array(all_omega)

        return_sta = {'force_mean': np.mean(all_forces, axis=0), 'force_std': np.std(all_forces, axis=0), 'cp_mean': np.mean(all_cps, axis=0),
                      'cp_std': np.std(all_cps, axis=0), 'position_mean': np.mean(all_position, axis=0),
                      'position_std': np.std(all_position, axis=0), 'rotation_mean': np.mean(all_rotation, axis=0),
                      'rotation_std': np.std(all_rotation, axis=0)}
        return return_sta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_d = self.data[idx]
        sta = self.data_statistics
        pm, pstd, fm, fst, cpm, cpstd = sta['position_mean'], sta['position_std'], sta['force_mean'], sta['force_std'], \
                                        sta['cp_mean'], sta['cp_std']
        initial_position, initial_rotation = torch.Tensor(cur_d['initial_position']), torch.Tensor(cur_d['initial_rotation'])
        target_position, target_rotation = torch.Tensor(cur_d['model_position'][0]), torch.Tensor(cur_d['model_rotation'][0])
        initial_env_state = NormEnvState(norm_or_denorm=True, position=initial_position, rotation=initial_rotation,
                                         position_mean=pm, position_std=pstd, velocity=None, velocity_mean=None,
                                         velocity_std=None, omega=None, omega_mean=None, omega_std=None)
        target_env_state = NormEnvState(norm_or_denorm=True, position=target_position, rotation=target_rotation,
                                        position_mean=pm, position_std=pstd, velocity=None, velocity_mean=None,
                                        velocity_std=None, omega=None, omega_mean=None, omega_std=None)
        force_tensor, cp_tensor = torch.Tensor(cur_d['force_applied'])[0], torch.Tensor(cur_d['model_contact_points'])[0]
        norm_force, norm_cp = norm_tensor(norm_or_denorm=True, tensor=force_tensor, mean_tensor=fm, std_tensor=fst), \
                              norm_tensor(norm_or_denorm=True, tensor=cp_tensor, mean_tensor=cpm, std_tensor=cpstd)

        initial_state_tensor, target_state_tensor = initial_env_state.toTensor().detach(), target_env_state.toTensor().detach()
        input_dict = {
            'norm_force': norm_force,
            'norm_contact_points': norm_cp,
            'state_dict': cur_d,
            'norm_state_tensor': initial_state_tensor,
        }

        labels = {
            'norm_state_tensor': target_state_tensor,
            'statistics': self.data_statistics,
        }

        return input_dict, labels
