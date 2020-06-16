import os
import torch
import torch.utils.data as data
from torch.nn.utils import rnn
import torchvision.transforms as transforms
from PIL import Image
import pdb
import random
import glob
from utils.data_io import read_from_json, read_from_pkl
from utils.environment_util import NormEnvState
from utils.tensor_utils import norm_tensor
from utils.custom_quaternion import *
import numpy as np
from IPython import embed


class NSDataset(data.Dataset):
    def __init__(self, obj_name, root_dir, train_val_rate, all_sequence, train_num=None, train=True, filter_d=True,
                 data_statistics=None):
        self.obj_name = obj_name
        self.root_dir = root_dir
        self.train_val_rate = train_val_rate
        self.filter_d = filter_d
        self.train = train
        self.all_sequence = all_sequence
        self.collate_fn = None if not self.all_sequence else lstm_collate_fn
        self.data_len = 0
        self.ele_location = []
        self.data = self.load_dataset()
        if train_num is not None and self.train and train_num < self.data_len:
            self.data_len = train_num
            self.ele_location = self.ele_location[:train_num]
        if self.train:
            self.data_statistics = self.cal_statistics()
        else:
            self.data_statistics = data_statistics
        print("total data num:", len(self.data))

    def filter_data(self, raw_data):
        final_data = []
        for idx, ele in enumerate(raw_data):
            pos_before, pos_after = torch.Tensor(ele['initial_position']), torch.Tensor(ele['model_position'][0])
            rot_before, rot_after = torch.Tensor(ele['initial_rotation']), torch.Tensor(ele['model_rotation'][0])
            if max(abs(pos_before - pos_after)) > 0.04:
                continue
            angle_before, angle_after = quaternion_to_euler_angle(rot_before), quaternion_to_euler_angle(rot_after)
            if max(abs(angle_after - angle_before)) > 20:
                continue
            final_data.append(ele)
            del pos_before, pos_after, rot_before, rot_after
        return final_data

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
        if self.filter_d:
            all_data = self.filter_data(all_data)
        split_ind = int(self.train_val_rate * len(all_data))
        final_d = all_data[:split_ind] if self.train else all_data[split_ind:]
        if not self.all_sequence:
            self.data_len = len(final_d)
        else:
            seq_idx = 0
            for idx, ele in enumerate(final_d):
                self.data_len += len(ele['model_rotation'])
                for jdx in range(len(ele['model_rotation'])):
                    cur_location = {'seq_id': idx, 'id_in_seq': jdx}
                    self.ele_location.append(cur_location)
                seq_idx += 1
            assert self.data_len == len(self.ele_location)
        return final_d

    def cal_statistics(self):
        print("Calculating data statistics.")
        all_forces, all_cps, all_position, all_rotation = [], [], [], []
        for idx, ele in enumerate(self.data):
            all_position.append(ele['initial_position'])
            all_rotation.append(ele['initial_rotation'])
            if not self.all_sequence:
                all_forces.append(ele['force_applied'][0])
                all_cps.append(ele['model_contact_points'][0])
                all_position.append(ele['model_position'][0])
                all_rotation.append(ele['model_rotation'][0])
            else:
                for seq_idx, _ in enumerate(ele['force_applied']):
                    all_forces.append(ele['force_applied'][seq_idx])
                    all_cps.append(ele['model_contact_points'][seq_idx])
                    all_position.append(ele['model_position'][seq_idx])
                    all_rotation.append(ele['model_rotation'][seq_idx])

        all_forces, all_cps, all_position, all_rotation = \
            np.array(all_forces), np.array(all_cps), np.array(all_position), np.array(all_rotation)

        return_sta = {'force_mean': np.mean(all_forces, axis=0), 'force_std': np.std(all_forces, axis=0), 'cp_mean': np.mean(all_cps, axis=0),
                      'cp_std': np.std(all_cps, axis=0), 'position_mean': np.mean(all_position, axis=0),
                      'position_std': np.std(all_position, axis=0), 'rotation_mean': np.mean(all_rotation, axis=0),
                      'rotation_std': np.std(all_rotation, axis=0)}
        return return_sta

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.all_sequence:
            cur_location = self.ele_location[idx]
            seq_id, id_in_seq = cur_location['seq_id'], cur_location['id_in_seq']
            cur_d = self.data[seq_id]
        else:
            cur_d = self.data[idx]
            seq_id, id_in_seq = None, 0
        sta = self.data_statistics
        pm, pstd, fm, fst, cpm, cpstd = sta['position_mean'], sta['position_std'], sta['force_mean'], sta['force_std'], \
                                        sta['cp_mean'], sta['cp_std']
        initial_position, initial_rotation = torch.Tensor(cur_d['initial_position']), torch.Tensor(cur_d['initial_rotation'])
        norm_force, norm_cp, norm_cur_state_tensor, norm_state_seq, norm_tar_state_tensor = None, None, None, [], None
        initial_env_state = NormEnvState(norm_or_denorm=True, position=initial_position, rotation=initial_rotation,
                                         position_mean=pm, position_std=pstd, velocity=None, velocity_mean=None,
                                         velocity_std=None, omega=None, omega_mean=None, omega_std=None)
        target_position, target_rotation = torch.Tensor(cur_d['model_position'][id_in_seq]), torch.Tensor(cur_d['model_rotation'][id_in_seq])
        target_env_state = NormEnvState(norm_or_denorm=True, position=target_position, rotation=target_rotation,
                                        position_mean=pm, position_std=pstd, velocity=None, velocity_mean=None,
                                        velocity_std=None, omega=None, omega_mean=None, omega_std=None)
        norm_tar_state_tensor = target_env_state.toTensor().detach()
        norm_cur_state_tensor = initial_env_state.toTensor().detach()
        if not self.all_sequence:
            norm_force, norm_cp = norm_tensor(norm_or_denorm=True, tensor=torch.Tensor(cur_d['force_applied'])[0], mean_tensor=fm, std_tensor=fst), \
                                  norm_tensor(norm_or_denorm=True, tensor=torch.Tensor(cur_d['model_contact_points'])[0], mean_tensor=cpm, std_tensor=cpstd)
        else:
            norm_state_seq = [initial_env_state.toTensor()]
            for i in range(id_in_seq - 1):
                cur_position, cur_rotation = torch.Tensor(cur_d['model_position'][i]), torch.Tensor(cur_d['model_rotation'][i])
                cur_env_state = NormEnvState(norm_or_denorm=True, position=cur_position, rotation=cur_rotation,
                                             position_mean=pm, position_std=pstd, velocity=None, velocity_mean=None,
                                             velocity_std=None, omega=None, omega_mean=None, omega_std=None)
                norm_state_seq.append(cur_env_state.toTensor())
                norm_cur_state_tensor = cur_env_state.toTensor()
            norm_state_seq = torch.stack(norm_state_seq)
            norm_force, norm_cp = norm_tensor(norm_or_denorm=True, tensor=torch.Tensor(cur_d['model_contact_points'])[id_in_seq],
                                              mean_tensor=fm, std_tensor=fst), \
                                  norm_tensor(norm_or_denorm=True, tensor=torch.Tensor(cur_d['model_contact_points'])[id_in_seq],
                                              mean_tensor=cpm, std_tensor=cpstd)
        input_dict = {
            'norm_force': norm_force,
            'norm_contact_points': norm_cp,
            # 'state_dict': cur_d,
            'norm_state_tensor': norm_cur_state_tensor,
            'norm_state_seq': norm_state_seq
        }

        labels = {
            'norm_state_tensor': norm_tar_state_tensor,
            'statistics': self.data_statistics,
        }

        return input_dict, labels


def lstm_collate_fn(batch_train_data):
    b_f, b_cp, b_state, b_seq_state, b_seq_len, b_ts = [], [], [], [], [], []
    stat = batch_train_data[0][1]['statistics']
    batch_train_data.sort(key=lambda data: len(data[0]['norm_state_seq']), reverse=True)
    for idx, ele in enumerate(batch_train_data):
        input_ele, label_ele = ele[0], ele[1]
        b_f.append(input_ele['norm_force'])
        b_cp.append(input_ele['norm_contact_points'])
        b_state.append(input_ele['norm_state_tensor'])
        b_seq_state.append(input_ele['norm_state_seq'])
        b_seq_len.append(len(input_ele['norm_state_seq']))
        b_ts.append(label_ele['norm_state_tensor'])
    b_seq_state = rnn.pad_sequence(b_seq_state, batch_first=True)
    final_input_dict = {'norm_force': torch.stack(b_f), 'norm_contact_points': torch.stack(b_cp),
                        'seq_len': b_seq_len, 'norm_state_tensor': torch.stack(b_state),
                        'norm_state_seq': b_seq_state}
    final_target_label = {'norm_state_tensor': torch.stack(b_ts), 'statistics': stat}
    return final_input_dict, final_target_label
