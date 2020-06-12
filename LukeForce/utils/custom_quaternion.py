from utils.transformations import euler_from_quaternion
import numpy as np
import torch
import math


def quaternion_to_euler_angle(quaternion, degree=True):
    '''
    Transform quaternion to euler angles.
    :param quaternion: in [w, x, y, z] format.
    :param degree: whether the result should be represented in degree.
    :return: euler angles.
    '''
    if type(quaternion) == torch.Tensor:
        quaternion = quaternion.cpu()
    euler_angle = euler_from_quaternion(quaternion, axes='rxyz')
    euler_angle = np.array(euler_angle)
    if degree:
        euler_angle = euler_angle / math.pi * 180
    return euler_angle


if __name__ == '__main__':
    import torch
    test_quat = torch.Tensor([0.224, 0.132, 0.414, 0.156])
    result_euler = quaternion_to_euler_angle(test_quat, degree=True)
    # [ x: -148.505844, y: 59.3978941, z: -162.9043098 ]
    print(result_euler)
