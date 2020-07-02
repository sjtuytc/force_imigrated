import torch
import time
import torch.nn as nn
from .base_model import BaseModel
from .ns_base_model import MLPNS, NSWithImageFeature
from utils.net_util import input_embedding_net, combine_block_w_do

from torchvision.models.resnet import resnet18
from solvers import metrics
from utils.environment_util import NoGradEnvState, nograd_envstate_from_tensor
from utils.projection_utils import get_keypoint_projection, get_all_objects_keypoint_tensors


def forward_resnet_feature(x, feature_extractor, train_res):
    def forward_x(ten):
        batch_size, seq_len, c, w, h = ten.shape
        ten = ten.view(batch_size * seq_len, c, w, h)
        ten = feature_extractor.conv1(ten)
        ten = feature_extractor.bn1(ten)
        ten = feature_extractor.relu(ten)
        ten = feature_extractor.maxpool(ten)

        ten = feature_extractor.layer1(ten)
        ten = feature_extractor.layer2(ten)
        ten = feature_extractor.layer3(ten)
        ten = feature_extractor.layer4(ten)

        ten = ten.view(batch_size, seq_len, 512, 7, 7)
        return ten
    if not train_res:
        feature_extractor.eval()
        with torch.no_grad():
            x = forward_x(x)
            x = x.detach()
    else:
        x = forward_x(x)
    return x


class ForcePredictor(nn.Module):
    def __init__(self, args):
        super(ForcePredictor, self).__init__()
        self.image_feature_size = 512
        self.object_feature_size = 512
        self.hidden_size = 512
        self.num_layers = 3
        self.sequence_length = args.sequence_length
        self.number_of_cp = args.number_of_cp
        self.use_gt_cp = args.use_gt_cp
        self.vis_grad = args.vis_grad
        self.train_res = args.train_res or self.vis_grad
        self.grad_value = None

        # force predictor networks.
        self.feature_extractor = resnet18(pretrained=args.pretrain)
        del self.feature_extractor.fc
        if not self.train_res:
            self.feature_extractor.eval()
        self.input_feature_size = self.object_feature_size
        self.cp_feature_size = self.number_of_cp * 3
        self.image_embed = combine_block_w_do(512, 64, args.dropout_ratio)
        self.contact_point_image_embed = combine_block_w_do(512, 64, args.dropout_ratio)

        input_object_embed_size = torch.Tensor([3 + 4, 100, self.object_feature_size])
        self.input_object_embed = input_embedding_net(input_object_embed_size.long().tolist(), dropout=args.dropout_ratio)
        self.contact_point_input_object_embed = input_embedding_net(input_object_embed_size.long().tolist(), dropout=args.dropout_ratio)

        state_embed_size = torch.Tensor([NoGradEnvState.total_size + self.cp_feature_size, 100, self.object_feature_size])
        self.state_embed = input_embedding_net(state_embed_size.long().tolist(), dropout=args.dropout_ratio)

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size + 64 * 7 * 7, hidden_size=self.hidden_size,
                                    batch_first=True, num_layers=self.num_layers)

        self.contact_point_encoder = nn.LSTM(input_size=self.hidden_size + 64 * 7 * 7, hidden_size=self.hidden_size,
                                             batch_first=True, num_layers=self.num_layers)
        contact_point_decoder_size = torch.Tensor([self.hidden_size, 100, (3) * self.number_of_cp])
        self.contact_point_decoder = input_embedding_net(contact_point_decoder_size.long().tolist(), dropout=args.dropout_ratio)

        self.lstm_decoder = nn.LSTMCell(input_size=self.hidden_size * 2, hidden_size=self.hidden_size)

        forces_directions_decoder_size = torch.Tensor([self.hidden_size, 100, (3) * self.number_of_cp])

        self.forces_directions_decoder = input_embedding_net(forces_directions_decoder_size.long().tolist(), dropout=args.dropout_ratio)

        assert args.batch_size == 1, 'have not been implemented yet, because of the environment'

        assert self.number_of_cp == 5  # for five fingers
        self.all_objects_keypoint_tensor = get_all_objects_keypoint_tensors(args.data)
        if args.gpu_ids != -1:
            for obj, val in self.all_objects_keypoint_tensor.items():
                self.all_objects_keypoint_tensor[obj] = val.cuda()

    def forward(self, input_dict, target):
        initial_position = input_dict['initial_position']
        initial_rotation = input_dict['initial_rotation']
        rgb = input_dict['rgb']
        batch_size, seq_len, c, w, h = rgb.shape
        object_name = input_dict['object_name']
        assert len(object_name) == 1  # only support one object
        object_name = object_name[0]
        image_features = forward_resnet_feature(x=rgb, feature_extractor=self.feature_extractor, train_res=self.train_res)

        # hooks to vis gradients
        if self.vis_grad:
            def grad_fn(grad):
                self.grad_value = grad.abs().mean()
            with torch.no_grad():
                if image_features.requires_grad:
                    handler1 = image_features.register_hook(grad_fn)

        if self.use_gt_cp:
            contact_points_prediction = input_dict['contact_points']
        else:
            # Contact point prediction tower
            image_features_contact_point = self.contact_point_image_embed(
                image_features.view(batch_size * seq_len, 512, 7, 7)).view(batch_size, seq_len, 64 * 7 * 7)
            initial_object_features_contact_point = \
                self.contact_point_input_object_embed(torch.cat([initial_position, initial_rotation], dim=-1))
            object_features_contact_point = initial_object_features_contact_point.unsqueeze(1). \
                repeat(1, self.sequence_length, 1)  # add a dimension for sequence length and then repeat that

            # Predict contact point
            input_embedded_contact_point = torch.cat([image_features_contact_point, object_features_contact_point], dim=-1)
            embedded_sequence_contact_point, (_, _) = self.contact_point_encoder(input_embedded_contact_point)
            contact_points_prediction = self.contact_point_decoder(embedded_sequence_contact_point).view(
                batch_size, seq_len, self.number_of_cp, 3)[:, -1, :, :]  # Predict contact point for each image

        # Force prediction tower
        image_features_force = self.image_embed(image_features.view(batch_size * seq_len, 512, 7, 7)).view(
            batch_size, seq_len, 64 * 7 * 7)
        initial_object_features_force = self.input_object_embed(torch.cat([initial_position, initial_rotation], dim=-1))
        object_features_force = initial_object_features_force.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # add a dimension for sequence length and then repeat that

        input_embedded_force = torch.cat([image_features_force, object_features_force], dim=-1)
        embedded_sequence_force, (hidden_force, cell_force) = self.lstm_encoder(input_embedded_force)

        last_hidden = hidden_force.view(self.num_layers, 1, 1, self.hidden_size)[-1, -1, :, :]
        # num_layers, num direction, batchsize, hidden and then take the last hidden layer
        last_cell = cell_force.view(self.num_layers, 1, 1, self.hidden_size)[-1, -1, :, :]
        # num_layers, num direction, batchsize, cell and then take the last hidden layer

        hn = last_hidden
        cn = last_cell

        initial_state = NoGradEnvState(object_name=object_name, rotation=initial_rotation[0], position=initial_position[0],
                                       velocity=None, omega=None)
        initial_state_tensor = initial_state.toTensorCoverName().unsqueeze(0)

        force_predictions = []
        for seq_ind in range(self.sequence_length - 1):
            contact_point_as_input = contact_points_prediction.view(1, 3 * 5)
            initial_state_and_cp = torch.cat([initial_state_tensor, contact_point_as_input], dim=-1)
            initial_frame_feature = self.state_embed(initial_state_and_cp)
            current_frame_feature = embedded_sequence_force[:, seq_ind + 1]
            input_lstm_cell = torch.cat([initial_frame_feature, current_frame_feature], dim=-1)
            (hn, cn) = self.lstm_decoder(input_lstm_cell, (hn, cn))
            force = self.forces_directions_decoder(hn)
            force_predictions.append(force)
        return contact_points_prediction, force_predictions


class NeuralForceSimulator(nn.Module):
    def __init__(self, args):
        super(NeuralForceSimulator, self).__init__()
        self.clean_force = True
        # neural force simulator
        self.use_image_feature = True
        self.vis_grad = args.vis_grad
        self.train_res = args.train_res or self.vis_grad
        self.hidden_size = 512
        self.num_layers = 3
        self.sequence_length = args.sequence_length
        self.object_feature_size = 512
        self.environment = args.instance_environment
        self.number_of_cp = args.number_of_cp
        self.feature_extractor = resnet18(pretrained=args.pretrain)
        del self.feature_extractor.fc
        if not self.train_res:
            self.feature_extractor.eval()
        if not self.use_image_feature:
            self.one_ns_layer = MLPNS(hidden_size=512, layer_norm=False)
        else:
            self.one_ns_layer = NSWithImageFeature(hidden_size=512, layer_norm=False, image_feature_dim=512)
        self.image_embed = combine_block_w_do(512, 64, args.dropout_ratio)
        input_object_embed_size = torch.Tensor([3 + 4, 100, self.object_feature_size])
        self.input_object_embed = input_embedding_net(input_object_embed_size.long().tolist(), dropout=args.dropout_ratio)
        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size + 64 * 7 * 7, hidden_size=self.hidden_size,
                                    batch_first=True, num_layers=self.num_layers)
        # self.ns_layer = {obj_name: MLPNS(hidden_size=64, layer_norm=False) for obj_name in self.all_obj_names}
        assert self.number_of_cp == 5  # for five fingers
        self.all_objects_keypoint_tensor = get_all_objects_keypoint_tensors(args.data)
        if args.gpu_ids != -1:
            for obj, val in self.all_objects_keypoint_tensor.items():
                self.all_objects_keypoint_tensor[obj] = val.cuda()

    def forward(self, input_dict, target, contact_points_prediction, force_predictions):
        initial_position = input_dict['initial_position']
        initial_rotation = input_dict['initial_rotation']
        rgb = input_dict['rgb']
        dev = rgb.device
        batch_size, seq_len, c, w, h = rgb.shape

        object_name = input_dict['object_name']
        assert len(object_name) == 1  # only support one object
        object_name = object_name[0]
        image_features = forward_resnet_feature(x=rgb, feature_extractor=self.feature_extractor, train_res=self.train_res)

        # hooks to vis gradients
        if self.vis_grad:
            def grad_fn(grad):
                self.grad_value = grad.abs().mean()
            with torch.no_grad():
                if image_features.requires_grad:
                    handler1 = image_features.register_hook(grad_fn)

        frame_features = self.image_embed(image_features.view(batch_size * seq_len, 512, 7, 7)).view(
            batch_size, seq_len, 64 * 7 * 7)
        initial_object_features_force = self.input_object_embed(torch.cat([initial_position, initial_rotation], dim=-1))
        object_features_force = initial_object_features_force.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # add a dimension for sequence length and then repeat that
        input_embedded = torch.cat([frame_features, object_features_force], dim=-1)
        ns_sequence_features, (hidden_force, cell_force) = self.lstm_encoder(input_embedded)

        # state initialization
        initial_state = NoGradEnvState(object_name=object_name, rotation=initial_rotation[0], position=initial_position[0],
                                       velocity=None, omega=None)
        initial_state_tensor = initial_state.toTensorCoverName().unsqueeze(0)
        ns_state_tensor = initial_state_tensor.clone()
        phy_env_state = nograd_envstate_from_tensor(object_name=object_name, env_tensor=initial_state_tensor[0],
                                                    clear_velocity=True)

        phy_positions, phy_rotations, ns_positions, ns_rotations, final_forces = [], [], [], [], []
        for seq_ind in range(self.sequence_length - 1):
            contact_point_as_input = contact_points_prediction.view(1, 3 * 5)
            current_frame_feature = ns_sequence_features[:, seq_ind + 1]

            # step physical simulator
            cur_force_pred = force_predictions[seq_ind]
            ns_env_state = nograd_envstate_from_tensor(object_name=object_name, env_tensor=ns_state_tensor[0],
                                                       clear_velocity=True)
            phy_env_state, succ_flags, force_locations, force_values = \
                self.environment.init_location_and_apply_force(forces=cur_force_pred[0].reshape(5, -1),
                                                               initial_state=ns_env_state,
                                                               list_of_contact_points=contact_point_as_input[0].reshape(5, -1),
                                                               no_grad=True, return_force_value=True)
            phy_state_tensor = phy_env_state.toTensorCoverName()
            phy_positions.append(phy_state_tensor[:3])
            phy_rotations.append(phy_state_tensor[3:7])

            # step neural force simulator
            # cur_obj_ns_layer = self.ns_layer[object_name]
            # predicted_state_tensor = cur_obj_ns_layer(ns_state_tensor, force, contact_point_as_input)
            if self.clean_force:
                force_values = [ele.to(dev) for ele in force_values]
                force_locations = [ele.to(dev) for ele in force_locations]
                cleaned_force, cleaned_cp = torch.stack(force_values).unsqueeze(0), \
                                            torch.stack(force_locations).unsqueeze(0)
            else:
                cleaned_force, cleaned_cp = cur_force_pred, contact_point_as_input

            if self.use_image_feature:
                predicted_state_tensor = self.one_ns_layer(ns_state_tensor[:, :7], cleaned_force, cleaned_cp,
                                                           current_frame_feature)
            else:
                predicted_state_tensor = self.one_ns_layer(ns_state_tensor, cleaned_force, cleaned_cp)

            # collect ns results
            ns_positions.append(predicted_state_tensor[0][:3])
            ns_rotations.append(predicted_state_tensor[0][3:7])
            final_forces.append(cleaned_force[0])

            # update to next state
            ns_state_tensor = predicted_state_tensor

        phy_positions = torch.stack(phy_positions).unsqueeze(0)
        phy_rotations = torch.stack(phy_rotations).unsqueeze(0)
        phy_kps = get_keypoint_projection(object_name, phy_positions, phy_rotations,
                                          self.all_objects_keypoint_tensor[object_name]).unsqueeze(0)
        ns_positions = torch.stack(ns_positions).unsqueeze(0)
        ns_rotations = torch.stack(ns_rotations).unsqueeze(0)
        final_forces = torch.stack(final_forces).unsqueeze(0)
        ns_kps = get_keypoint_projection(object_name, ns_positions, ns_rotations,
                                         self.all_objects_keypoint_tensor[object_name]).unsqueeze(0)

        contact_points_prediction = contact_points_prediction.unsqueeze(1).repeat(1, seq_len, 1, 1)

        output_dict = {
            'phy_position': phy_positions,
            'phy_rotation': phy_rotations,
            'phy_keypoints': phy_kps,
            'ns_position': ns_positions,
            'ns_rotation': ns_rotations,
            'ns_keypoints': ns_kps,
            'contact_points': contact_points_prediction,
            'force': final_forces,
        }
        return output_dict


# a model supports batch inference parallelly.
class SeperateFPAndNS(BaseModel):
    metric = [
        metrics.CPMetric,
        metrics.StateGroundingMetric,
        metrics.ForceGroundingMetric,
        metrics.ForcePredictionMetric,
    ]

    def __init__(self, args):
        super(SeperateFPAndNS, self).__init__(args)
        self.loss_function = args.loss
        self.number_of_cp = args.number_of_cp
        self.gpu_ids = args.gpu_ids
        self.all_obj_names = args.object_list

        # configs w.r.t. two losses
        self.joint_two_losses = args.joint_two_losses
        self.loss1_or_loss2 = None
        if args.loss1_w < 0.00001:
            self.loss1_or_loss2 = False   # update loss2 only
        elif args.loss2_w < 0.00001:
            self.loss1_or_loss2 = True    # update loss1 only
        self.ns_optim, self.fp_optim = None, None

        # see gradients for debugging
        self.vis_grad = args.vis_grad
        self.grad_vis = None

        self.train_res = args.train_res or self.vis_grad

        self.fp = ForcePredictor(args)
        self.ns = NeuralForceSimulator(args=args)

    def loss(self, args):
        return self.loss_function(args)

    def forward(self, input_dict, target):
        # first forward, only optimize neural force simulator
        cp_prediction, force_prediction = self.fp(input_dict, target)
        output_dict = self.ns(input_dict, target, cp_prediction, force_prediction)
        target['object_name'] = input_dict['object_name']
        return output_dict, target

    def optimizer(self):
        self.fp_optim = torch.optim.Adam(self.fp.parameters(), lr=self.base_lr)
        self.ns_optim = torch.optim.Adam(self.ns.parameters(), lr=self.base_lr)
        return None

    def step_optimizer(self, loss1_or_loss2):
        if self.loss1_or_loss2 is not None:
            loss1_or_loss2 = self.loss1_or_loss2
        if loss1_or_loss2:  # True for update ns, False for update force.
            self.ns_optim.step()
        else:
            self.fp_optim.step()
        self.zero_grad()

    def set_learning_rate(self, lr):
        if self.ns_optim is not None:
            for param_group in self.ns_optim.param_groups:
                param_group['lr'] = lr
        if self.fp_optim is not None:
            for param_group in self.fp_optim.param_groups:
                param_group['lr'] = lr

    def cuda(self, device=None):
        self._apply(lambda t: t.cuda(device))
        # for one_obj in self.ns_layer:
        #     self.ns_layer[one_obj].cuda()
        return self

    def to(self, *args, **kwargs):
        raise NotImplementedError("Please use .cuda() instead.")
