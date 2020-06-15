from datasets.ns_dataset import NSDataset
import torch
import os
from utils.tensor_utils import norm_tensor
from utils.transformations import euler_from_quaternion
from utils.visualization_util import vis_state
from utils.arg_parser import parse_args
from environments.np_physics_env import NpPhysicsEnv
import matplotlib.pyplot as plt
from IPython import embed


def get_all_property(dataset, name):
    all_data_point = []
    # rot_mean = torch.Tensor(dataset.data_statistics['rotation_mean'])
    pos_mean = torch.Tensor(dataset.data_statistics['position_mean'])
    pos_std = torch.Tensor(dataset.data_statistics['position_std'])
    # rot_std = torch.Tensor(dataset.data_statistics['rotation_std'])
    for idx in range(len(dataset)):
        input_dict, labels = dataset[idx]
        if name == 'before_angle':
            before_tensor = input_dict['norm_state_tensor'][3:7]
            before_angle1 = euler_from_quaternion(before_tensor, axes='rxyz')[0] / 3.14 * 180
            before_angle2 = euler_from_quaternion(before_tensor, axes='rxyz')[1] / 3.14 * 180
            before_angle3 = euler_from_quaternion(before_tensor, axes='rxyz')[2] / 3.14 * 180
            all_data_point += [before_angle1, before_angle2, before_angle3]
        elif name == 'position':
            before_pos = norm_tensor(norm_or_denorm=False, tensor=input_dict['norm_state_tensor'][:3],
                                     mean_tensor=pos_mean, std_tensor=pos_std)
            all_data_point += [before_pos[0], before_pos[1], before_pos[2]]
        elif name == 'rot_diff':
            before_rot = input_dict['norm_state_tensor'][3:7]
            before_angle = euler_from_quaternion(before_rot, axes='rxyz')[0] / 3.14 * 180
            after_rot = labels['norm_state_tensor'][3:7]
            after_angle = euler_from_quaternion(after_rot, axes='rxyz')[0] / 3.14 * 180
            diff_rot = after_angle - before_angle
            if abs(diff_rot) > 40:
                embed()
            all_data_point += [diff_rot]
        elif name == 'pos_diff':
            before_pos = norm_tensor(norm_or_denorm=False, tensor=input_dict['norm_state_tensor'][:3], mean_tensor=pos_mean,
                                     std_tensor=pos_std)
            after_pos = norm_tensor(norm_or_denorm=False, tensor=labels['norm_state_tensor'][:3], mean_tensor=pos_mean,
                                    std_tensor=pos_std)
            diff_pos = after_pos - before_pos
            all_data_point += [diff_pos[0], diff_pos[1], diff_pos[2]]
    return all_data_point


def plot_distribution():
    root_dir = "NSDatasetV6/"
    train_d = NSDataset(obj_name='019_pitcher_base', root_dir=root_dir, train_val_rate=0.9, all_sequence=True,
                        train=True, data_statistics=None, filter_d=True)
    fig, ax = plt.subplots()
    one_data_points = get_all_property(train_d, 'before_angle')
    ax.hist(one_data_points, bins=10, label='angle of input state (degree)')
    plt.title('angle distribution')
    cur_p = os.path.join("debug/angle_distribution.png")
    plt.legend()
    print("Result image is saved at:", cur_p)
    plt.savefig(cur_p, dpi=1000)
    fig, ax = plt.subplots()
    one_data_points = get_all_property(train_d, 'position')
    ax.hist(one_data_points, bins=10, label='position of input state (m)')
    plt.title('position distribution')
    cur_p = os.path.join("debug/position_distribution.png")
    plt.legend()
    print("Result image is saved at:", cur_p)
    plt.savefig(cur_p, dpi=1000)
    fig, ax = plt.subplots()
    one_data_points = get_all_property(train_d, 'pos_diff')
    ax.hist(one_data_points, bins=10, label='position difference between consecutive states (m)')
    plt.title('position difference distribution')
    cur_p = os.path.join("debug/position_diff_distribution.png")
    plt.legend()
    print("Result image is saved at:", cur_p)
    plt.savefig(cur_p, dpi=1000)

    fig, ax = plt.subplots()
    one_data_points = get_all_property(train_d, 'rot_diff')
    ax.hist(one_data_points, bins=10, label='rotation difference between consecutive states (degree)')
    plt.title('rotation difference distribution')
    cur_p = os.path.join("debug/rotation_diff_distribution.png")
    plt.legend()
    print("Result image is saved at:", cur_p)
    plt.savefig(cur_p, dpi=1000)
    embed()


def visualize_one_d(visualize_env, obj_name, dataset, idx):
    input_dict, labels = dataset[idx]
    rot_mean = torch.Tensor(dataset.data_statistics['rotation_mean'])
    pos_mean = torch.Tensor(dataset.data_statistics['position_mean'])
    pos_std = torch.Tensor(dataset.data_statistics['position_std'])
    rot_std = torch.Tensor(dataset.data_statistics['rotation_std'])
    before_position = norm_tensor(norm_or_denorm=False, tensor=input_dict['norm_state_tensor'][:3], mean_tensor=pos_mean,
                                  std_tensor=pos_std)
    before_rot = norm_tensor(norm_or_denorm=False, tensor=input_dict['norm_state_tensor'][3:7], mean_tensor=rot_mean,
                             std_tensor=rot_std)
    after_position = norm_tensor(norm_or_denorm=False, tensor=labels['norm_state_tensor'][:3], mean_tensor=pos_mean,
                                 std_tensor=pos_std)
    after_rot = norm_tensor(norm_or_denorm=False, tensor=labels['norm_state_tensor'][3:7], mean_tensor=rot_mean,
                            std_tensor=rot_std)
    print(before_rot, after_rot)
    embed()
    vis_state(vis_env=visualize_env, obj_name=obj_name, position=before_position, rotation=before_rot, image_name='before_state',
              save_folder='debug/', verbose=True)
    vis_state(vis_env=visualize_env, obj_name=obj_name, position=after_position, rotation=after_rot, image_name='after_state',
              save_folder='debug/', verbose=True)


if __name__ == '__main__':
    # root_dir = "NSDatasetV5/"
    # args = parse_args(log_info=False, save_log=False)
    # obj_name = '019_pitcher_base'
    # object_path = os.path.join(args.data, 'objects_16k', obj_name, 'google_16k', 'textured.urdf')
    # vis_env = NpPhysicsEnv(render=args.render, object_name=obj_name, object_path=object_path, gravity=args.gravity,
    #                         debug=args.debug, number_of_cp=args.number_of_cp, fps=args.fps, force_multiplier=args.force_multiplier,
    #                         force_h=args.force_h, state_h=args.state_h, qualitative_size=args.qualitative_size, workers=args.workers,
    #                         gpu_ids=args.gpu_ids)
    # vis_env.reset()
    # train_d = NSDataset(obj_name=obj_name, root_dir=root_dir, train_val_rate=0.9, train=True,
    #                 data_statistics=None)
    # visualize_one_d(vis_env, obj_name, train_d, 183)
    plot_distribution()
