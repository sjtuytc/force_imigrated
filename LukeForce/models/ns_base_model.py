import torch
import torch.nn as nn
from .base_model import BaseModel
from solvers import metrics
import torch.nn.functional as F


class MlpLayer(nn.Module):
    def __init__(self, input_d, output_d):
        super(MlpLayer, self).__init__()
        self.fc1 = nn.Linear(input_d, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.output_fc = nn.Linear(512, output_d)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.output_fc(x)
        return x


def get_denorm_state_tensor(state_ten, stat):
    pos_mean, pos_std, rot_mean, rot_std, vel_mean, vel_std, omg_mean, omg_std = \
        stat['position_mean'], stat['position_std'], stat['rotation_mean'], stat['rotation_std'], stat['velocity_mean'], \
        stat['velocity_std'], stat['omega_mean'], stat['omega_std']
    state_ten[:, :3] = state_ten[:, :3] * pos_std + pos_mean
    state_ten[:, 7:10] = state_ten[:, 7:10] * vel_std + vel_mean
    state_ten[:, 10:] = state_ten[:, 10:] * omg_std + omg_mean
    return state_ten


class NSBaseModel(BaseModel):
    metric = [metrics.StateMetric]

    def __init__(self, args, ):
        super(NSBaseModel, self).__init__(args)
        self.loss_function = args.loss
        self.force_feature_size, self.state_feature_size, self.cp_feature_size = 64, 64, 64
        self.state_tensor_dim, self.force_tensor_dim, self.cp_tensor_dim = 13, 15, 15
        self.state_encoder = MlpLayer(input_d=self.state_tensor_dim, output_d=self.state_feature_size)
        self.force_encoder = MlpLayer(input_d=self.force_tensor_dim, output_d=self.force_feature_size)
        self.cp_encoder = MlpLayer(input_d=self.cp_tensor_dim, output_d=self.cp_feature_size)
        self.force_decoder = MlpLayer(input_d=self.state_feature_size + self.force_feature_size + self.cp_feature_size,
                                      output_d=self.state_tensor_dim)
        self.number_of_cp = args.number_of_cp
        self.sequence_length = args.sequence_length
        self.gpu_ids = args.gpu_ids

    def loss(self, args):
        return self.loss_function(args)

    def forward(self, input_d, target_d):
        forces, contact_points, state_tensor = input_d['norm_force'], input_d['norm_contact_points'], \
                                               input_d['norm_state_tensor']
        batch_size = forces.shape[0]
        forces, contact_points, state_tensor = forces.reshape(batch_size, -1), contact_points.reshape(batch_size, -1)\
            , state_tensor.reshape(batch_size, -1)
        force_feature = self.force_encoder(forces)
        state_feature = self.state_encoder(state_tensor)
        cp_feature = self.cp_encoder(contact_points)
        fused_feature = torch.cat([force_feature, state_feature, cp_feature], dim=-1)
        predict_state = self.force_decoder(fused_feature)
        sta = {one_key: target_d['statistics'][one_key].squeeze() for one_key in target_d['statistics']}
        target_d['norm_state_tensor'] = target_d['norm_state_tensor'].reshape(batch_size, -1)
        target_d['denorm_state_tensor'] = get_denorm_state_tensor(state_ten=target_d['norm_state_tensor'], stat=sta)
        denorm_predict_state = get_denorm_state_tensor(state_ten=predict_state.clone(), stat=sta)
        output_d = {
            'norm_state_tensor': predict_state,
            'denorm_state_tensor': denorm_predict_state
        }
        return output_d, target_d

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
