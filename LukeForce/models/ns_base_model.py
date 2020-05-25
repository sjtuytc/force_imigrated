import torch
import torch.nn as nn
from utils.environment_util import EnvState
from utils.projection_utils import get_keypoint_projection, get_all_objects_keypoint_tensors
from .base_model import BaseModel
# from utils.net_util import input_embedding_net, EnvWHumanCpFiniteDiffFast, combine_block_w_do
# from torchvision.models.resnet import resnet18
from solvers import metrics


class MlpLayer(nn.Module):
    def __init__(self, input_d, hidden_d, output_d):
        super(MlpLayer, self).__init__()
        self.fc1 = nn.Linear(input_d, hidden_d)
        self.fc2 = nn.Linear(hidden_d, hidden_d)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_d, output_d)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


class NSBaseModel(BaseModel):
    metric = []

    def __init__(self, args, ):
        super(NSBaseModel, self).__init__(args)
        self.loss_function = args.loss
        self.hidden_size = 512
        self.force_feature_size, self.state_feature_size, self.cp_feature_size = 64, 64, 64
        self.state_tensor_dim, self.force_tensor_dim, self.cp_tensor_dim = 14, 15, 15
        self.state_encoder = MlpLayer(input_d=self.state_tensor_dim, hidden_d=self.hidden_size, output_d=self.state_feature_size)
        self.force_encoder = MlpLayer(input_d=self.force_tensor_dim, hidden_d=self.hidden_size, output_d=self.force_feature_size)
        self.cp_encoder = MlpLayer(input_d=self.cp_tensor_dim, hidden_d=self.hidden_size, output_d=self.cp_feature_size)
        self.force_decoder = MlpLayer(input_d=self.state_feature_size + self.force_feature_size + self.cp_feature_size,
                                      hidden_d=self.hidden_size, output_d=self.state_tensor_dim)
        self.number_of_cp = args.number_of_cp
        self.sequence_length = args.sequence_length
        self.gpu_ids = args.gpu_ids

    def loss(self, args):
        return self.loss_function(args)

    def forward(self, input_d, target_d):
        forces, contact_points, state_tensor = input_d['force'], input_d['contact_points'], input_d['state_tensor']
        batch_size = forces.shape[0]
        forces, contact_points, state_tensor = forces.reshape(batch_size, -1), contact_points.reshape(batch_size, -1)\
            , state_tensor.reshape(batch_size, -1)
        force_feature = self.force_encoder(forces)
        state_feature = self.state_encoder(state_tensor)
        cp_feature = self.cp_encoder(contact_points)
        fused_feature = torch.cat([force_feature, state_feature, cp_feature], dim=-1)
        predict_state = self.force_decoder(fused_feature)
        target_d['state_tensor'] = target_d['state_tensor'].reshape(batch_size, -1)
        output_d = {
            'state_tensor': predict_state
        }
        return output_d, target_d

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
