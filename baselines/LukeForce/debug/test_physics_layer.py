import os
import torch
import random
import torch.nn as nn
from utils.net_util import CPGradientLayer, BatchCPGradientLayer
from utils.arg_parser import parse_args
from environments.physics_env import PhysicsEnv
from environments.subproc_physics_env import SubprocPhysicsEnv
from utils.environment_util import EnvState


class PhysicEnvTester(nn.Module):
    def __init__(self, args, object_paths, object_name, context, nproc, nenvs, parallel=True):
        super(PhysicEnvTester, self).__init__()
        if parallel:
            self.environment = SubprocPhysicsEnv(args=args, object_paths=object_paths,
                                                 object_name=object_name, context=context, nproc=nproc, nenvs=nenvs)
            self.phy_gradient_layer = BatchCPGradientLayer
        else:
            raise NotImplementedError("Unparallel version is not supported now.")
        self.environment.reset()

    def forward(self, env_state_tensor, force_tensor, contact_point_tensor):
        env_state, force_success, force_applied = \
            self.phy_gradient_layer.apply(self.environment, env_state_tensor, force_tensor,
                                          contact_point_tensor)
        return env_state, force_success, force_applied


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_dir = args.data
    object_list = args.object_list

    object_paths = {obj: os.path.join(root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for
                    obj in object_list}


    test_obj = '005_tomato_soup_can'
    nproc = 11
    nenvs = 44
    print("Creating %d envs in %d processes." %(nenvs, nproc))
    test_m = PhysicEnvTester(args=args, object_path=object_paths[test_obj], object_name=test_obj, context='spawn',
                             nproc=nproc, nenvs=nenvs, parallel=True)
    force_data = torch.Tensor([[-0.0047, -0.0841, -0.0801] for i in range(5)])
    initial_state = EnvState(object_name='005_tomato_soup_can', position=(-0.2421,  0.0213,  0.9691),
                             rotation=(0.3126, -0.5557,  0.4300, -0.6392),
                             velocity=(0.0, 0.0, 0.0), omega=(0., 0., 0.))
    list_of_contact_points = torch.Tensor([[-0.0934, -0.0214,  0.0495], [-0.0651, -0.0909, -0.0848],
                                           [-0.0042,  0.0523, -0.0092], [0.0637, -0.0939, -0.0402],
                                           [0.0107,  0.0885,  0.0101]])
    env_state, force_success, force_applied = test_m(env_state_tensor=initial_state.toTensor(),
                                                     force_tensor=force_data.flatten(),
                                                     contact_point_tensor=list_of_contact_points)



if __name__ == '__main__':
    main()
