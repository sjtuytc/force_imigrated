from utils.arg_parser import parse_args
import random
import torch
import logging
import os
import time
from environments.physics_env import PhysicsEnv
from utils.environment_util import ForceValOnly, EnvState


def init_env_and_apply_force(args, object_path, object_name, forces, initial_state, object_num, list_of_contact_points):
    phy_env = PhysicsEnv(args=args, object_path=object_path, object_name=object_name)
    phy_env.reset()
    current_state, list_of_force_success, list_of_force_location = \
        phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=object_num,
                                              list_of_contact_points=list_of_contact_points)

    phy_env.close()
    print("Finish testing in one function and closing envs.")
    return


def test_original():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                     obj in object_list}

    test_obj = '005_tomato_soup_can'
    phy_env = PhysicsEnv(render=args.render, object_name=test_obj,
                         object_path=object_paths[test_obj],
                         gravity=args.gravity, debug=args.debug,
                         number_of_cp=args.number_of_cp, gpu_ids=args.gpu_ids,
                         fps=args.fps, force_multiplier=args.force_multiplier,
                         force_h=args.force_h, state_h=args.state_h,
                         qualitative_size=args.qualitative_size, workers=0)
    phy_env.reset()

    # generate test data.
    forces = [ForceValOnly(force=(-0.0047, -0.0841, -0.0801)) for i in range(5)]
    initial_state = EnvState(object_name='005_tomato_soup_can', position=(-0.2421,  0.0213,  0.9691),
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
    for k in range(44*2000):
        current_state, list_of_force_success, list_of_force_location = \
            phy_env.init_location_and_apply_force(forces=forces, initial_state=initial_state, object_num=None,
                                                  list_of_contact_points=list_of_contact_points)
    print("Consuming time: ", time.time() - bt)


def test_one_function():
    args = parse_args(log_info=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                    obj in object_list}

    test_obj = '005_tomato_soup_can'
    forces = [ForceValOnly(force=(-0.0047, -0.0841, -0.0801)) for i in range(5)]
    initial_state = EnvState(object_name='005_tomato_soup_can', position=torch.Tensor([-0.2421,  0.0213,  0.9691]),
                             rotation=torch.Tensor([0.3126, -0.5557,  0.4300, -0.6392]),
                             velocity=torch.Tensor([0.0, 0.0, 0.0]), omega=torch.Tensor([0., 0., 0.]))
    list_of_contact_points = torch.Tensor([[-0.0934, -0.0214,  0.0495],
                                           [-0.0651, -0.0909, -0.0848],
                                           [-0.0042,  0.0523, -0.0092],
                                           [0.0637, -0.0939, -0.0402],
                                           [0.0107,  0.0885,  0.0101]])

    init_env_and_apply_force(args=args, object_path=object_paths[test_obj], object_name=test_obj, forces=forces,
                             initial_state=initial_state, object_num=None,
                             list_of_contact_points=list_of_contact_points)


if __name__ == '__main__':
    print("Test in one function.")
    test_original()
