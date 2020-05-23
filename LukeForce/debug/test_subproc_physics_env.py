from utils.arg_parser import parse_args
import random
import logging
import os
import time
from environments.subproc_physics_env import SubprocPhysicsEnv


def test_original():
    args = parse_args()

    random.seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                    obj in object_list}

    nproc = 11
    nenvs = 44
    phy_env = SubprocPhysicsEnv(args=args, object_paths=object_paths, context='spawn',
                                nproc=nproc, nenvs=nenvs)

    phy_env.reset()

    # generate test data.
    force_data = [(-0.0047, -0.0841, -0.0801) for i in range(5)]
    state_data = {'object_name': '005_tomato_soup_can', 'position': (-0.2421,  0.0213,  0.9691),
                  'rotation': (0.3126, -0.5557,  0.4300, -0.6392), 'velocity': (0.0, 0.0, 0.0),
                  'omega': (0., 0., 0.)}
    # Initially, one cannot use the list version as input.
    list_of_contact_points = [[-0.0934, -0.0214,  0.0495], [-0.0651, -0.0909, -0.0848], [-0.0042,  0.0523, -0.0092],
                              [0.0637, -0.0939, -0.0402], [0.0107,  0.0885,  0.0101]]
    one_data = {'forces': force_data, 'initial_state': state_data, 'object_num': None, 'list_of_contact_points':
                list_of_contact_points}

    batch_test_data = [one_data for i in range(nenvs)]

    print("Test infer time.")
    bt = time.time()
    for i in range(1024):
        phy_env.batch_init_locations_and_apply_force(batch_data=batch_test_data)
    total_time = time.time() - bt
    print("Consuming time: ", total_time)
    print("Time per call: ", total_time / 2000 / nenvs)


if __name__ == '__main__':
    test_original()
