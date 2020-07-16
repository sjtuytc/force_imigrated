from utils.arg_parser import parse_args
import random
import torch
import logging
import os
import time
from environments.np_physics_env import NpPhysicsEnv
from environments.physics_env import PhysicsEnv
from utils.environment_util import NpForceValOnly, NpEnvState, ForceValOnly, EnvState
from utils.constants import GRAVITY_VALUE
from utils.custom_quaternion import quaternion_to_euler_angle, w_first_to_w_last
from IPython import embed


def test_speed():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                     obj in object_list}

    test_obj = '005_tomato_soup_can'
    phy_env = NpPhysicsEnv(render=args.render, object_name=test_obj, object_path=object_paths[test_obj],
                            gravity=args.gravity, debug=args.debug, number_of_cp=args.number_of_cp, fps=args.fps,
                           force_multiplier=args.force_multiplier,
                            force_h=args.force_h, state_h=args.state_h, qualitative_size=args.qualitative_size,
                           workers=0)
    phy_env.reset()

    # generate test data.
    forces = [NpForceValOnly(force=(-0.0047, -0.0841, -0.0801)) for i in range(5)]
    initial_state = NpEnvState(object_name='005_tomato_soup_can', position=(-0.2421,  0.0213,  0.9691),
                                rotation=(0.3126, -0.5557,  0.4300, -0.6392),
                                velocity=(0.0, 0.0, 0.0), omega=(0., 0., 0.))

    # Initially, one cannot use the list version as input.
    list_of_contact_points = [[-0.0934, -0.0214,  0.0495], [-0.0651, -0.0909, -0.0848], [-0.0042,  0.0523, -0.0092],
                              [0.0637, -0.0939, -0.0402], [0.0107,  0.0885,  0.0101]]
    # list_of_contact_points = torch.Tensor([[-0.0934, -0.0214,  0.0495],
    #                                        [-0.0651, -0.0909, -0.0848],
    #                                        [-0.0042,  0.0523, -0.0092],
    #                                        [0.0637, -0.0939, -0.0402],
    #                                        [0.0107,  0.0885,  0.0101]])
    print("Test infer time.")
    bt = time.time()
    for k in range(2000):
        current_state, list_of_force_success, list_of_force_location = \
            phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=None,
                                                  list_of_contact_points=list_of_contact_points)
    total_t = time.time() - bt
    print("Consuming time: ", total_t)
    print("Time per call:", total_t / 2000)


def compare_to_original():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                    obj in object_list}

    test_obj = '005_tomato_soup_can'
    np_phy_env = NpPhysicsEnv(render=args.render, object_name=test_obj, object_path=object_paths[test_obj],
                                gravity=args.gravity, debug=args.debug, number_of_cp=args.number_of_cp, fps=args.fps,
                                force_multiplier=args.force_multiplier,
                                force_h=args.force_h, state_h=args.state_h, qualitative_size=args.qualitative_size,
                                workers=0)
    np_phy_env.reset()

    # generate test data.
    forces = [ForceValOnly(force=(-0.0047, -0.0841, -0.0801)).tolist() for i in range(5)]
    initial_state = EnvState(object_name='005_tomato_soup_can', position=(-0.2421,  0.0213,  0.9691),
                               rotation=(0.3126, -0.5557,  0.4300, -0.6392),
                               velocity=(0.0, 0.0, 0.0), omega=(0., 0., 0.)).to_dict()

    list_of_contact_points = [[-0.0934, -0.0214,  0.0495], [-0.0651, -0.0909, -0.0848], [-0.0042,  0.0523, -0.0092],
                              [0.0637, -0.0939, -0.0402], [0.0107,  0.0885,  0.0101]]
    current_state, list_of_force_success, list_of_force_location = \
        np_phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=None,
                                                     list_of_contact_points=list_of_contact_points)

    phy_env = PhysicsEnv(render=args.render, object_name=test_obj, object_path=object_paths[test_obj], gravity=args.gravity,
                         debug=args.debug, number_of_cp=args.number_of_cp, fps=args.fps, force_multiplier=args.force_multiplier,
                         force_h=args.force_h, state_h=args.state_h, qualitative_size=args.qualitative_size,
                         workers=0, gpu_ids=0)
    phy_env.reset()
    new_state, new_force_success, new_force_location = \
        phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=None,
                                              list_of_contact_points=list_of_contact_points)

    print("Np env:", current_state, list_of_force_success, list_of_force_location)
    print("Ori env:", new_state, new_force_success, new_force_location)


def try_one_spec(args, obj_name, obj_path, forces, cps, initial_state):
    phy_env = PhysicsEnv(render=args.render, object_name=obj_name, object_path=obj_path, gravity=False,
                         debug=args.debug, number_of_cp=args.number_of_cp, fps=args.fps, force_multiplier=args.force_multiplier,
                         force_h=args.force_h, state_h=args.state_h, qualitative_size=args.qualitative_size,
                         workers=0, gpu_ids=0)
    phy_env.reset()
    new_state, new_force_success, new_force_location, cleaned_force_values = \
        phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=None,
                                              list_of_contact_points=cps, return_force_value=True)
    pos_diff = new_state.position - initial_state.position
    total_raw_force = sum(forces).cpu() * phy_env.force_multiplier
    total_cleaned_force = sum(cleaned_force_values).cpu() * phy_env.force_multiplier
    print("Raw force/pos diff:", total_raw_force / pos_diff)
    # print("Cleaned force/pos diff:", total_cleaned_force / pos_diff)

    # phy_env = PhysicsEnv(render=args.render, object_name=test_obj, object_path=object_paths[test_obj], gravity=True,
    #                      debug=args.debug, number_of_cp=args.number_of_cp, fps=args.fps, force_multiplier=args.force_multiplier,
    #                      force_h=args.force_h, state_h=args.state_h, qualitative_size=args.qualitative_size,
    #                      workers=0, gpu_ids=0)
    # phy_env.reset()
    # new_state, new_force_success, new_force_location, cleaned_force_values = \
    #     phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=None,
    #                                           list_of_contact_points=list_of_contact_points, return_force_value=True)
    initial_rot, new_rot = w_first_to_w_last(initial_state.rotation.detach()), \
                           w_first_to_w_last(new_state.rotation.detach())
    quat_diff = torch.Tensor(phy_env.get_euler_angles(new_rot)) - torch.Tensor(phy_env.get_euler_angles(initial_rot))
    position_vector = torch.stack(new_force_location)
    torques = torch.cross(position_vector.cpu(), torch.Tensor(forces) * phy_env.force_multiplier)
    total_torque = sum(torques).cpu()
    print("Raw torque/quat diff:", total_torque / quat_diff.cpu())

    print("Total torque:", total_torque)
    new_state = phy_env.init_location_and_apply_torque(torques=total_torque, initial_state=initial_state,
                                                       object_num=None)
    new_rot = w_first_to_w_last(new_state.rotation.detach())
    quat_diff = torch.Tensor(phy_env.get_euler_angles(new_rot)) - torch.Tensor(phy_env.get_euler_angles(initial_rot))
    print("After apply torque directly, raw torque/quat diff:", total_torque / quat_diff.cpu())
    new_state = phy_env.init_location_and_apply_torque(torques=total_torque, initial_state=new_state,
                                                       object_num=None)
    embed()
    # pos_diff = new_state.position - initial_state.position
    # total_raw_force_with_grav = force_obj.force * 5 * phy_env.force_multiplier + GRAVITY_VALUE
    # total_cleaned_force_with_grav = sum(cleaned_force_values).cpu() * phy_env.force_multiplier + GRAVITY_VALUE
    # print("Raw force/pos prediction:", total_raw_force_with_grav / pos_diff)
    # print("Cleaned force/pos prediction:", total_cleaned_force_with_grav / pos_diff)


def test_phy_env():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                    obj in object_list}

    test_obj = '021_bleach_cleanser'

    # generate test data.
    forces = torch.Tensor([[-0.0750,  0.0945, -0.0002],
                           [-0.0191,  0.1162, -0.0168],
                           [ 0.0008, -0.0522, -0.0968],
                           [-0.1309, -0.0555, -0.0008],
                           [-0.0618,  0.0685,  0.0590]])
    initial_state = EnvState(object_name=test_obj, position=(-0.2421,  0.0213,  0.9691),
                             rotation=(0.7615,  0.2002, -0.1179, -0.6051),
                             velocity=(0.0, 0.0, 0.0), omega=(0., 0., 0.))

    list_of_contact_points = torch.Tensor([[-0.0042,  0.0363,  0.1326],
                                          [-0.0165,  0.0426,  0.0519],
                                          [0.0193,  0.0353,  0.0753],
                                          [0.0200,  0.0355,  0.0590],
                                          [0.0184,  0.0352,  0.0843]])

    # try_one_spec(args, obj_name=test_obj, obj_path=object_paths[test_obj], forces=forces, cps=list_of_contact_points,
    #              initial_state=initial_state)
    new_forces = torch.Tensor([[-0.0750,  0.0945, 0.1002],
                               [0.0191,  0.1162, -0.0168],
                               [ 0.0008, -0.0522, -0.0968],
                               [-0.0109, -0.0555, -0.0008],
                               [-0.0618,  0.0685,  0.1590]])
    try_one_spec(args, obj_name=test_obj, obj_path=object_paths[test_obj], forces=new_forces, cps=list_of_contact_points,
                 initial_state=initial_state)


if __name__ == '__main__':
    print("Test in one function.")
    test_phy_env()
