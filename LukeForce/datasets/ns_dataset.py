import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pdb
import random
import glob
from utils.data_io import read_from_json
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
        files = glob.glob(os.path.join(self.root_dir, '*.json'))
        assert len(files) > 0, 'dataset load failed'

        one_d = None
        for one_file in files:
            if self.obj_name in one_file:
                one_d = read_from_json(full_path=one_file, verbose=False)
                break
        assert one_d is not None, "data file of target obj is not found!"
        if type(one_d) == dict and self.obj_name in one_d:
            final_d = one_d[self.obj_name]
        else:
            final_d = one_d
        random.seed(0)
        random.shuffle(final_d)
        split_ind = int(self.train_val_rate * len(final_d))
        final_d = final_d[:split_ind] if self.train else final_d[split_ind:]
        return final_d

    def cal_statistics(self):
        print("Calculating data statistics.")
        all_forces, all_cps, all_position, all_rotation, all_velocity, all_omega = [], [], [], [], [], []
        for idx, ele in enumerate(self.data):
            ci, co = ele['input'], ele['output']
            forces, cps, position, rotation, velocity, omega = \
                co['force_values'], co['loc'], ci['initial_state']['position'], \
                ci['initial_state']['rotation'], ci['initial_state']['velocity'], ci['initial_state']['omega']
            all_forces.append(forces)
            all_cps.append(cps)
            all_position.append(position)
            all_rotation.append(rotation)
            all_velocity.append(velocity)
            all_omega.append(omega)
        all_forces, all_cps, all_position, all_rotation, all_velocity, all_omega = np.array(all_forces), np.array(all_cps), \
            np.array(all_position), np.array(all_rotation), np.array(all_velocity), np.array(all_omega)

        return_sta = {'force_mean': np.mean(all_forces, axis=0), 'force_std': np.std(all_forces, axis=0), 'cp_mean': np.mean(all_cps, axis=0),
                      'cp_std': np.std(all_cps, axis=0), 'position_mean': np.mean(all_position, axis=0),
                      'position_std': np.std(all_position, axis=0), 'rotation_mean': np.mean(all_rotation, axis=0),
                      'rotation_std': np.std(all_rotation, axis=0), 'velocity_mean': np.mean(all_velocity, axis=0),
                      'velocity_std': np.std(all_velocity, axis=0), 'omega_mean': np.mean(all_omega, axis=0),
                      'omega_std': np.std(all_omega, axis=0)}
        return return_sta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_d = self.data[idx]
        input_d, output_d = cur_d['input'], cur_d['output']
        isd, tsd = input_d['initial_state'], output_d['state']
        sta = self.data_statistics
        pm, pstd, vm, vst, fm, fst, omm, omstd, cpm, cpstd = sta['position_mean'], sta['position_std'], \
                                                              sta['velocity_mean'], sta['velocity_std'], \
                                                              sta['force_mean'], sta['force_std'], sta['omega_mean'],\
                                                              sta['omega_std'], sta['cp_mean'], sta['cp_std']
        initial_env_state = NormEnvState(norm_or_denorm=True, position=isd['position'], rotation=isd['rotation'],
                                         position_mean=pm, position_std=pstd, velocity=isd['velocity'], velocity_mean=vm,
                                         velocity_std=vst, omega=isd['omega'], omega_mean=omm, omega_std=omstd)
        target_env_state = NormEnvState(norm_or_denorm=True, position=tsd['position'], rotation=tsd['rotation'],
                                        position_mean=pm, position_std=pstd, velocity=tsd['velocity'], velocity_mean=vm,
                                        velocity_std=vst, omega=tsd['omega'], omega_mean=omm, omega_std=omstd)
        force_tensor, cp_tensor = torch.Tensor(output_d['force_values']), torch.Tensor(output_d['loc'])
        norm_force, norm_cp = norm_tensor(norm_or_denorm=True, tensor=force_tensor, mean_tensor=fm, std_tensor=fst), \
                              norm_tensor(norm_or_denorm=True, tensor=cp_tensor, mean_tensor=cpm, std_tensor=cpstd)

        input_dict = {
            'norm_force': norm_force,
            'norm_contact_points': norm_cp,
            'state_dict': isd,
            'norm_state_tensor': initial_env_state.toTensor().detach(),
        }

        labels = {
            'norm_state_tensor': target_env_state.toTensor().detach(),
            'statistics': self.data_statistics,
        }

        return input_dict, labels
