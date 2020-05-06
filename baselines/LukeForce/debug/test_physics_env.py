from utils.arg_parser import parse_args
import random
import torch
import logging
import os
import time
from environments.np_physics_env import NpPhysicsEnv
from utils.environment_util import NpForceValOnly, NpEnvState


def test_original():
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


if __name__ == '__main__':
    print("Test in one function.")
    test_original()
