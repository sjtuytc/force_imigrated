import torch
from utils.quaternion_util import get_quaternion_distance
from utils.custom_quaternion import quaternion_to_euler_angle
from utils.constants import DEFAULT_IMAGE_SIZE

class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.sum = None
        self.count = 0

    def update(self, val, n=1):
        with torch.no_grad():
            if self.val is None:
                self.val = val
                self.sum = self.val * n
            else:
                self.val = val
                self.sum += val * n
            self.count += n

    @property
    def avg(self):
        with torch.no_grad():
            return self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return 'Avg({}),Ct({}),Sum({})'.format(self.avg, self.count, self.sum)


class BaseMetric:

    def record_output(self, output, target):
        raise Exception('record_output is not implemented')

    def report(self):
        raise Exception('report is not implemented')

    def final_report(self):
        return self.report()

    def average(self):
        raise Exception('avearge is not implemented')


'''
Idea is the same as the evaluation code in http://trajnet.stanford.edu/result.php?cid=1
'''


class TrajectoryDistanceMetric(BaseMetric):
    def __init__(self, args):
        super(TrajectoryDistanceMetric, self).__init__()
        self.sequence_length = args.sequence_length
        self.meter = {
            'trajectory_dist_3d': AverageMeter(),
        }
        self.number_of_cp = args.number_of_cp

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        output_position = output['position']
        target_position = target['position']
        assert output_position.shape == target_position.shape

        batch_size = output_position.shape[0]
        diff_l2_dist = (output_position - target_position).norm(dim=-1).mean(0)  # norm over last dimension (x,y,z) and mean over first dimension(batch size)

        self.meter['trajectory_dist_3d'].update(diff_l2_dist, n=batch_size)

    def report(self):
        return 'TrajectoryDistanceMetric:{}'.format({k: x.mean() for k, x in self.average().items()}).replace('\n', '; ')


class CPMetric(BaseMetric):

    def __init__(self, args):
        super(CPMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {'time_{}'.format(time): AverageMeter() for time in range(self.sequence_length)}
        self.meter['average_cp'] = AverageMeter()
        for obj in self.object_list:
            self.meter[obj] = AverageMeter()
        # assert args.mode == 'test' or args.mode == 'testtrain'
        self.istraining = args.mode == 'train'

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):

        output_contact_points = output['contact_points']
        target_contact_points = target['contact_points']
        batch_size, seq_len, num_cp, dims = output_contact_points.shape

        target_contact_points = target_contact_points.unsqueeze(1).repeat(1, seq_len, 1, 1)
        assert output_contact_points.shape == target_contact_points.shape

        loss_cp_prediction = torch.abs(output_contact_points - target_contact_points).norm(dim=-1)
        loss_cp_prediction = loss_cp_prediction.view(batch_size, seq_len, num_cp).mean(dim=-1)

        for batch_ind in range(batch_size):
            obj_name = target['object_name'][batch_ind]
            self.meter[obj_name].update(loss_cp_prediction[batch_ind].mean().detach(), 1)

        loss_cp_prediction = loss_cp_prediction.mean(dim=0)

        average_cp = output_contact_points.mean(dim=1)
        not_repeated_target = target['contact_points']
        assert average_cp.shape == not_repeated_target.shape
        loss_average_cp = torch.abs(average_cp - not_repeated_target).norm(dim=-1).mean()
        for time in range(seq_len):
            self.meter['time_{}'.format(time)].update(loss_cp_prediction[time], batch_size)

        self.meter['average_cp'].update(loss_average_cp, batch_size)

    def report(self):
        return 'CPMetric:{}'.format((self.meter['average_cp'].avg)).replace('\n', '; ')


class ObjRotationMetric(BaseMetric):

    def __init__(self, args):
        super(ObjRotationMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {obj: AverageMeter() for obj in self.object_list}
        self.meter['overall'] = AverageMeter()
        self.distance_function = get_quaternion_distance

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        output_rotation = output['rotation']
        target_rotation = target['rotation']
        object_name = target['object_name']
        assert len(object_name) == 1
        object_name = object_name[0]

        assert output_rotation.shape == target_rotation.shape
        batch_size = output_rotation.shape[0]

        dist = self.distance_function(output_rotation, target_rotation).mean()

        self.meter[object_name].update(dist.detach(), batch_size)
        self.meter['overall'].update(dist.detach(), batch_size)

    def report(self):
        return 'ObjRotationMetric:{}'.format(self.meter['overall'].avg).replace('\n', '; ')


def l2_dist(output, target):
    last_dim = output.shape[-1]
    output = output.view(-1, last_dim)
    target = target.view(-1, last_dim)
    return torch.nn.PairwiseDistance()(output, target).mean()


class ObjPositionMetric(BaseMetric):

    def __init__(self, args):
        super(ObjPositionMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {obj: AverageMeter() for obj in self.object_list}
        self.meter['overall'] = AverageMeter()
        self.distance_function = l2_dist

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        output_position = output['position']
        target_position = target['position']
        object_name = target['object_name']
        assert len(object_name) == 1
        object_name = object_name[0]

        assert output_position.shape == target_position.shape
        batch_size = output_position.shape[0]

        dist = self.distance_function(output_position, target_position)

        self.meter[object_name].update(dist.detach(), batch_size)
        self.meter['overall'].update(dist.detach(), batch_size)

    def report(self):
        return 'ObjPositionMetric:{}'.format(self.meter['overall'].avg).replace('\n', '; ')


def cal_kp_dis(output_kp, target_kp):
    # Note: this is different from original implementation.
    kp_range = int(1.5 * max(DEFAULT_IMAGE_SIZE.cpu().tolist()))
    assert output_kp.shape == target_kp.shape
    if target_kp.device != output_kp.device:
        target_kp = target_kp.to(output_kp.device)
    output_kp = output_kp.clamp(-kp_range, kp_range)
    target_kp = target_kp.clamp(-kp_range, kp_range)
    mask = target_kp <= 1e-10
    not_masked = ~ (mask.sum(dim=-1) > 0)
    if not_masked.sum().item() > 0:
        diff = torch.abs(output_kp[not_masked] - target_kp[not_masked]).norm(dim=-1)
        keypoint_loss = diff.mean()
    else:
        keypoint_loss = None
    return keypoint_loss


class ObjKeypointMetric(BaseMetric):

    def __init__(self, args):
        super(ObjKeypointMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {obj: AverageMeter() for obj in self.object_list}
        self.meter['overall'] = AverageMeter()

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        output_keypoints = output['keypoints']
        target_keypoints = target['keypoints']
        keypoint_loss = cal_kp_dis(output_kp=output_keypoints, target_kp=target_keypoints)
        batch_size = 1
        object_name = target['object_name']
        assert len(object_name) == 1
        object_name = object_name[0]
        self.meter[object_name].update(keypoint_loss, batch_size)
        self.meter['overall'].update(keypoint_loss, batch_size)

    def report(self):
        return 'ObjKeypointMetric:{}'.format((self.meter['overall'].avg)).replace('\n', '; ')


class StateGroundingMetric(BaseMetric):
    def __init__(self, args):
        super(StateGroundingMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {obj: AverageMeter() for obj in self.object_list}
        self.meter['overall'] = AverageMeter()

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        ns_kp = output['ns_keypoints']
        gt_kp = target['keypoints']
        keypoint_loss = cal_kp_dis(output_kp=ns_kp, target_kp=gt_kp)
        if keypoint_loss is not None:
            batch_size = 1
            object_name = target['object_name']
            assert len(object_name) == 1
            object_name = object_name[0]
            self.meter[object_name].update(keypoint_loss, batch_size)
            self.meter['overall'].update(keypoint_loss, batch_size)

    def report(self):
        return 'StateGroundingMetric:{}'.format((self.meter['overall'].avg)).replace('\n', '; ')


class ForceGroundingMetric(BaseMetric):
    def __init__(self, args):
        super(ForceGroundingMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {obj: AverageMeter() for obj in self.object_list}
        self.meter['overall'] = AverageMeter()

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        ns_kp = output['ns_keypoints']
        phy_kp = output['phy_keypoints']
        keypoint_loss = cal_kp_dis(output_kp=ns_kp, target_kp=phy_kp)
        if keypoint_loss is not None:
            batch_size = 1
            object_name = target['object_name']
            assert len(object_name) == 1
            object_name = object_name[0]
            self.meter[object_name].update(keypoint_loss, batch_size)
            self.meter['overall'].update(keypoint_loss, batch_size)

    def report(self):
        return 'ForceGroundingMetric:{}'.format((self.meter['overall'].avg)).replace('\n', '; ')


class ForcePredictionMetric(BaseMetric):
    def __init__(self, args):
        super(ForcePredictionMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {obj: AverageMeter() for obj in self.object_list}
        self.meter['overall'] = AverageMeter()

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        gt_kp = target['keypoints']
        phy_kp = output['phy_keypoints']
        keypoint_loss = cal_kp_dis(output_kp=phy_kp, target_kp=gt_kp)
        if keypoint_loss is not None:
            batch_size = 1
            object_name = target['object_name']
            assert len(object_name) == 1
            object_name = object_name[0]
            self.meter[object_name].update(keypoint_loss, batch_size)
            self.meter['overall'].update(keypoint_loss, batch_size)

    def report(self):
        return 'ForcePredictionMetric:{}'.format((self.meter['overall'].avg)).replace('\n', '; ')


def cal_euler_diffrence(quat_a, quat_b):
    euler_a, euler_b = quaternion_to_euler_angle(quat_a), quaternion_to_euler_angle(quat_b)
    diff_result = torch.abs(torch.Tensor(euler_a) - torch.Tensor(euler_b)).mean()
    return diff_result


class StateMetric(BaseMetric):
    # deprecating, used in old ns version.
    def __init__(self, args):
        super(StateMetric, self).__init__()
        self.object_list = args.object_list
        self.sequence_length = args.sequence_length
        self.meter = {'avg_position': AverageMeter(), 'avg_rotation': AverageMeter(), 'avg_omega': AverageMeter(),
                      'avg_speed': AverageMeter(), 'base_pos': AverageMeter(), 'base_rot': AverageMeter()}

        # assert args.mode == 'test' or args.mode == 'testtrain'
        self.istraining = args.mode == 'train'
        self.predict_speed = args.predict_speed

    def average(self):
        return {k: self.meter[k].avg for k in self.meter}

    def val(self):
        return {k: self.meter[k].val for k in self.meter}

    def record_output(self, output, target):
        output_state_tensor = output['denorm_state_tensor'].detach()
        target_state_tensor = target['denorm_state_tensor'].detach()
        input_state_tensor = target['denorm_input_state'].detach()
        batch_size = output_state_tensor.shape[0]

        pos_loss = torch.abs(output_state_tensor[:, :3] - target_state_tensor[:, :3]).mean()
        base_pos_diff = torch.abs(input_state_tensor[:, :3] - target_state_tensor[:, :3]).mean()

        ang_dis, base_ang_dis = 0, 0
        for i in range(batch_size):
            ang_dis += cal_euler_diffrence(output_state_tensor[i][3:7].cpu(), target_state_tensor[i][3:7].cpu())
            base_ang_dis += cal_euler_diffrence(input_state_tensor[i][3:7].cpu(), target_state_tensor[i][3:7].cpu())
        ang_dis = ang_dis / batch_size
        base_ang_dis = base_ang_dis / batch_size
        if self.predict_speed:
            vel_dis = torch.abs(output_state_tensor[:, 7:10] - target_state_tensor[:, 7:10]).mean()
            omg_dis = torch.abs(output_state_tensor[:, 10:] - target_state_tensor[:, 10:]).mean()
        else:
            vel_dis = -1
            omg_dis = -1
        self.meter['avg_position'].update(pos_loss, batch_size)
        self.meter['avg_rotation'].update(ang_dis, batch_size)
        self.meter['avg_omega'].update(omg_dis, batch_size)
        self.meter['avg_speed'].update(vel_dis, batch_size)
        self.meter['base_pos'].update(base_pos_diff, batch_size)
        self.meter['base_rot'].update(base_ang_dis, batch_size)

    def report(self):
        report_str = 'Position Dis: {:.3f}; '.format(self.meter['avg_position'].avg)
        report_str += 'Rotation Dis: {:.3f}; '.format(self.meter['avg_rotation'].avg)
        report_str += 'Pos Copy Dis: {:.3f}; '.format(self.meter['base_pos'].avg)
        report_str += 'Rot Copy Dis: {:.3f}; '.format(self.meter['base_rot'].avg)
        report_str += 'Omega Dis: {:.3f}; '.format(self.meter['avg_omega'].avg)
        report_str += 'Vel Dis: {:.3f}.'.format(self.meter['avg_speed'].avg)
        return report_str
