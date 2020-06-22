import torch
import torch.nn as nn
from utils.constants import DEFAULT_IMAGE_SIZE
from IPython import embed

__all__ = [
    'KeypointProjectionLoss',
    'KPProjectionCPPredictionLoss',
    'CPPredictionLoss',
    'ForceRegressionLoss',
    'StateEstimationLoss',
    'JointNSProjectionLoss',
]

variables = locals()


class BasicLossFunction(nn.Module):
    def __init__(self):
        super(BasicLossFunction, self).__init__()

    @property
    def local_loss_dict(self):

        module_attributes = self._modules.keys()
        result = self._local_loss_dict
        for mod in module_attributes:
            attr = self.__getattr__(mod)
            if issubclass(type(attr), BasicLossFunction):
                result.update(attr.local_loss_dict)
        return result

    def calc_and_update_total_loss(self, loss_dict, batch_size):
        total = 0
        for k in loss_dict:
            self._local_loss_dict[k] = (loss_dict[k], batch_size)
            total += loss_dict[k] * self.weights_for_each_loss[k]
        return total


class StateEstimationLoss(BasicLossFunction):
    # deprecating
    def __init__(self, args):
        super(BasicLossFunction, self).__init__()
        self.loss_fn = nn.MSELoss()
        self._local_loss_dict = {
            'state_prediction': None,
        }
        self.weights_for_each_loss = {
            'state_prediction': 1.,
        }
        self.predict_speed = args.predict_speed
        self.balance = args.balance

    def forward(self, output_t, target_t):
        output_state = output_t['norm_state_tensor']
        bs = output_state.shape[0]
        target_state = target_t['norm_state_tensor']
        if self.predict_speed:
            loss_state_estimation_value = self.loss_fn(output_state, target_state)
        else:
            pos_loss = self.loss_fn(output_state[:, :3], target_state[:, :3])
            rot_loss = self.loss_fn(output_state[:, 3:7], target_state[:, 3:7])
            loss_state_estimation_value = pos_loss + self.balance * rot_loss
            loss_state_estimation_value /= (1 + self.balance)
        loss_dict = {
            'state_prediction': loss_state_estimation_value,
        }
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size=bs)
        return total_loss


class CPPredictionLoss(BasicLossFunction):
    def __init__(self, args):
        super(CPPredictionLoss, self).__init__()
        self.loss_contact_point_loss = nn.SmoothL1Loss()
        self._local_loss_dict = {
            'contact_point_projection': None,
        }
        self.weights_for_each_loss = {
            'contact_point_projection': 1.,
        }

    def forward(self, output, target):
        output_contact_points = output['contact_points']
        target_contact_points = target['contact_points']
        batch_size, seq_len, num_cp, dims = output_contact_points.shape
        target_contact_points = target_contact_points.unsqueeze(1).repeat(1, seq_len, 1, 1)
        assert output_contact_points.shape == target_contact_points.shape

        loss_cp_prediction_value = self.loss_contact_point_loss(output_contact_points, target_contact_points)

        loss_dict = {
            'contact_point_projection': loss_cp_prediction_value,
        }

        batch_size = output_contact_points.shape[0]

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class ForceRegressionLoss(BasicLossFunction):
    def __init__(self, args):
        super(ForceRegressionLoss, self).__init__()
        self.loss_force_loss = nn.MSELoss()
        self._local_loss_dict = {
            'force_projection': None,
        }
        self.weights_for_each_loss = {
            'force_projection': 1.,
        }

    def forward(self, output, target):
        output_forces = output['forces']
        target_forces = target['forces']

        assert output_forces.shape == target_forces.shape

        loss_force_value = self.loss_force_loss(output_forces, target_forces)

        loss_dict = {
            'force_projection': loss_force_value,
        }

        batch_size = output_forces.shape[0]

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


def cal_kp_loss(output_keypoints, target_keypoints, loss_fn, default_img_size=DEFAULT_IMAGE_SIZE):
    assert output_keypoints.shape == target_keypoints.shape
    if output_keypoints.device != default_img_size.device:
        default_img_size = default_img_size.to(output_keypoints.device)
    if target_keypoints.device != output_keypoints.device:
        target_keypoints = target_keypoints.to(output_keypoints.device)
    output_keypoints = output_keypoints / default_img_size
    target_keypoints = target_keypoints / default_img_size

    output_keypoints = torch.clamp(output_keypoints, -5, 5)
    number_of_elements = target_keypoints.numel()
    not_masked_target = target_keypoints > 1e-10
    number_of_unmasked_elements = not_masked_target.sum().item()

    if number_of_unmasked_elements > 0:
        keypoint_projection = loss_fn(output_keypoints[not_masked_target], target_keypoints[not_masked_target])
        keypoint_projection = keypoint_projection * float(number_of_unmasked_elements) / number_of_elements  # Normalization
    else:  # Just a dummy thing
        keypoint_projection = torch.tensor(0.0, requires_grad=True, device=output_keypoints.device)
    return keypoint_projection


class KeypointProjectionLoss(BasicLossFunction):
    def __init__(self, args):
        super(KeypointProjectionLoss, self).__init__()
        self.loss_keypoint_loss = nn.SmoothL1Loss()
        self._local_loss_dict = {
            'keypoint_projection': None,
        }
        self.weights_for_each_loss = {
            'keypoint_projection': 1.,
        }
        self.default_image_size = DEFAULT_IMAGE_SIZE

    def forward(self, output, target):
        output_keypoints = output['keypoints']
        target_keypoints = target['keypoints']
        kp_loss = cal_kp_loss(output_keypoints=output_keypoints, target_keypoints=target_keypoints,
                              loss_fn=self.loss_keypoint_loss, default_img_size=self.default_image_size)
        loss_dict = {
            'keypoint_projection': kp_loss,
        }

        # summarize losses
        batch_size = output_keypoints.shape[0]
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class KPProjectionCPPredictionLoss(BasicLossFunction):
    def __init__(self, args):
        super(KPProjectionCPPredictionLoss, self).__init__()
        self.loss_kp_projection = KeypointProjectionLoss(args)
        self.loss_cp_prediction = CPPredictionLoss(args)
        self._local_loss_dict = {
            'loss_kp_projection': None,
            'loss_cp_prediction': None,
        }
        self.weights_for_each_loss = {
            'loss_kp_projection': 1.,
            'loss_cp_prediction': 5.,
        }

    def forward(self, output, target):
        loss_kp_projection_value = self.loss_kp_projection(output, target)
        loss_cp_prediction_value = self.loss_cp_prediction(output, target)

        loss_dict = {
            'loss_kp_projection': loss_kp_projection_value,
            'loss_cp_prediction': loss_cp_prediction_value,
        }
        batch_size = output['contact_points'].shape[0]

        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss


class JointNSProjectionLoss(BasicLossFunction):
    def __init__(self, args):
        super(JointNSProjectionLoss, self).__init__()
        self.loss_keypoint_loss = nn.SmoothL1Loss()
        self.default_image_size = DEFAULT_IMAGE_SIZE
        self.loss_cp_prediction = CPPredictionLoss(args)
        self._local_loss_dict = {
            'loss1_state_grounding': None,
            'loss2_force_grouding': None,
            'loss_cp_prediction': None,
        }
        self.weights_for_each_loss = {
            'loss1_state_grounding': args.loss1_w,
            'loss2_force_grouding': args.loss2_w,
            'loss_cp_prediction': 5.,
        }
        self.force_or_ns = None
        if args.loss1_w < 0.0001:
            self.force_or_ns = True   # update force only
        elif args.loss2_w < 0.0001:
            self.force_or_ns = False    # update ns only

    def forward(self, output, target):
        force_or_ns = target['force_or_ns']
        if self.force_or_ns is not None:
            force_or_ns = self.force_or_ns
        loss_cp_prediction_value = self.loss_cp_prediction(output, target)
        gt_kps = target['keypoints']
        phy_kps = output['phy_keypoints']
        ns_kps = output['ns_keypoints']
        if force_or_ns is None:
            loss1_state_grounding_value = cal_kp_loss(ns_kps, gt_kps, self.loss_keypoint_loss,
                                                      default_img_size=self.default_image_size)
            loss2_force_grounding_value = cal_kp_loss(ns_kps, phy_kps, self.loss_keypoint_loss,
                                                      default_img_size=self.default_image_size)
            loss_dict = {
                'loss1_state_grounding': loss1_state_grounding_value,
                'loss2_force_grouding': loss2_force_grounding_value,
                'loss_cp_prediction': loss_cp_prediction_value,
            }
        elif not force_or_ns:
            loss1_state_grounding_value = cal_kp_loss(ns_kps, gt_kps, self.loss_keypoint_loss,
                                                      default_img_size=self.default_image_size)
            loss_dict = {
                'loss1_state_grounding': loss1_state_grounding_value,
                'loss_cp_prediction': loss_cp_prediction_value,
            }
        else:
            loss2_force_grounding_value = cal_kp_loss(ns_kps, phy_kps, self.loss_keypoint_loss,
                                                      default_img_size=self.default_image_size)
            loss_dict = {
                'loss2_force_grouding': loss2_force_grounding_value,
                'loss_cp_prediction': loss_cp_prediction_value,
            }

        # summarize losses
        batch_size = output['contact_points'].shape[0]
        total_loss = self.calc_and_update_total_loss(loss_dict, batch_size)

        return total_loss
