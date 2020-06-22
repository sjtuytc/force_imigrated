import torch
import numpy as np
import torch.nn.functional as F
from utils.constants import ALL_OBJECTS
from utils.tensor_utils import norm_tensor

REGISTERED_OBJECTS = ALL_OBJECTS


def convert_obj_name_to_tensor(object_name):
    return torch.Tensor([REGISTERED_OBJECTS.index(object_name)]).float()


def convert_tensor_to_obj_name(object_name_tensor):
    object_ind = object_name_tensor.item()
    int_object_ind = int(object_ind)
    assert (int_object_ind - object_ind) <= 0.01
    assert int_object_ind < len(REGISTERED_OBJECTS)
    return REGISTERED_OBJECTS[int_object_ind]


# env state objects.
def build_env_state_from_dict(env_state_dict):
    # one needs to convert the env state to dict and vice versa because env_state is not piclable.
    return EnvState(object_name=env_state_dict['object_name'], position=env_state_dict['position'],
                    rotation=env_state_dict['rotation'], velocity=env_state_dict['velocity'],
                    omega=env_state_dict['omega'])


def build_np_env_state_from_dict(env_state_dict):
    # one needs to convert the env state to dict and vice versa because env_state is not piclable.
    return NpEnvState(object_name=env_state_dict['object_name'], position=env_state_dict['position'],
                        rotation=env_state_dict['rotation'], velocity=env_state_dict['velocity'],
                        omega=env_state_dict['omega'])


class NpEnvState:
    '''
    Use numpy to accellerate gradient computation, deprecating now.
    '''
    size = [3, 4, 3, 3, 1]
    total_size = sum(size)
    OBJECT_TYPE_INDEX = total_size - 1

    def __init__(self, object_name, position, rotation, velocity=None, omega=None):
        if velocity is None:
            velocity = np.array([0., 0., 0.])
        if omega is None:
            omega = np.array([0., 0., 0.])
        position, rotation, velocity, omega = \
            np.array(position), np.array(rotation), np.array(velocity), np.array(omega)
        assert len(position) == 3 and len(rotation) == 4 and len(velocity) == 3 and len(omega) == 3

        # ((1+0.01*z)/(1+2z*0.01+0.01^2)) -> function of diff of (w,x,y,z) - (w,x,y,z+eps)
        # rotation = F.normalize(rotation.expand_dims(axis=0)).squeeze(0)
        expanded_rotation = np.expand_dims(rotation, axis=0)
        rotation = (expanded_rotation / np.linalg.norm(expanded_rotation)).squeeze(axis=0)
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.omega = omega
        self.object_name = object_name

    def clone(self):
        return NpEnvState(object_name=self.object_name, position=self.position, rotation=self.rotation, velocity=self.velocity, omega=self.omega)

    def __str__(self):
        return 'object_name:{},position:{},rotation:{},velocity:{},omega:{}'.format(self.object_name, self.position, self.rotation, self.velocity, self.omega)

    def to_dict(self):
        return {'object_name': self.object_name, 'position': self.position.tolist(), 'rotation': self.rotation.tolist(),
                'velocity': self.velocity.tolist(), 'omega': self.omega.tolist()}

    def toTensor(self, device):
        tensor_position, tensor_rotation, tensor_velocity, tensor_omega = torch.Tensor(self.position).to(device), torch.Tensor(self.rotation).to(device), \
                                                                          torch.Tensor(self.velocity).to(device), torch.Tensor(self.omega).to(device)
        assert self.object_name in REGISTERED_OBJECTS
        object_name_tensor = convert_obj_name_to_tensor(self.object_name)
        object_name_tensor = object_name_tensor.to(device)
        result_tensor = torch.cat([tensor_position, tensor_rotation, tensor_velocity, tensor_omega, object_name_tensor], dim=-1)
        assert result_tensor.shape[0] == NpEnvState.total_size
        return result_tensor

    @staticmethod
    def fromTensor(tensor):
        assert tensor.shape[0] == NpEnvState.total_size and len(tensor.shape) == 1, "Shape does not match when creating NpEnvState"
        position = tensor[0:3].detach().cpu()
        rotation = tensor[3:7].detach().cpu()
        velocity = tensor[7:10].detach().cpu()
        omega = tensor[10:13].detach().cpu()
        assert 0 <= tensor[EnvState.OBJECT_TYPE_INDEX] < len(REGISTERED_OBJECTS)
        object_name = convert_tensor_to_obj_name(tensor[EnvState.OBJECT_TYPE_INDEX])
        return NpEnvState(object_name, position, rotation, velocity, omega)


def nograd_envstate_from_tensor(object_name, env_tensor, clear_velocity):
    if clear_velocity:
        return_state = NoGradEnvState(object_name=object_name, position=env_tensor[:3], rotation=env_tensor[3:7],
                                      velocity=None, omega=None, device=env_tensor.device)
    else:
        return_state = NoGradEnvState(object_name=object_name, position=env_tensor[:3], rotation=env_tensor[3:7],
                                      velocity=env_tensor[7:10], omega=env_tensor[10:13], device=env_tensor.device)
    return return_state


class NoGradEnvState:
    size = [3, 4, 3, 3, 1]
    total_size = sum(size)
    OBJECT_TYPE_INDEX = total_size - 1

    def __init__(self, object_name, position, rotation, velocity=None, omega=None, device=None):
        if velocity is None:
            velocity = torch.tensor([0., 0., 0.], device=position.device, )
        if omega is None:
            omega = torch.tensor([0., 0., 0.], device=position.device,)

        assert len(position) == 3 and len(rotation) == 4 and len(velocity) == 3 and len(omega) == 3

        [position, rotation, velocity, omega] = [convert_to_tensor(x, require_grad=False)
                                                 for x in [position, rotation, velocity, omega]]
        if device is not None:
            [position, rotation, velocity, omega] = [x.to(device) for x in [position, rotation, velocity, omega]]

        rotation = F.normalize(rotation.unsqueeze(0)).squeeze(0)

        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.omega = omega
        self.object_name = object_name

    def toTensorCoverName(self):
        assert type(self.position) == torch.Tensor
        assert type(self.rotation) == torch.Tensor
        assert type(self.velocity) == torch.Tensor
        assert type(self.omega) == torch.Tensor
        assert self.object_name in REGISTERED_OBJECTS
        object_name_tensor = torch.Tensor([-1.0]).float().to(self.position.device)
        tensor = torch.cat([self.position, self.rotation, self.velocity, self.omega, object_name_tensor], dim=-1)
        assert tensor.shape[0] == EnvState.total_size
        return tensor

    def __str__(self):
        return 'object_name:{},position:{},rotation:{},velocity:{},omega:{}'.format(self.object_name, self.position, self.rotation, self.velocity, self.omega)

    def cuda_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cuda() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def cpu_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cpu() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def to_dict(self):
        return {'object_name': self.object_name, 'position': self.position.tolist(), 'rotation': self.rotation.tolist(),
                'velocity': self.velocity.tolist(), 'omega': self.omega.tolist()}


class EnvState:
    size = [3, 4, 3, 3, 1]
    total_size = sum(size)
    OBJECT_TYPE_INDEX = total_size - 1

    def __init__(self, object_name, position, rotation, velocity=None, omega=None, device=None):
        if velocity is None:
            velocity = torch.tensor([0., 0., 0.], device=position.device, requires_grad=True)
        if omega is None:
            omega = torch.tensor([0., 0., 0.], device=position.device, requires_grad=True)

        assert len(position) == 3 and len(rotation) == 4 and len(velocity) == 3 and len(omega) == 3

        [position, rotation, velocity, omega] = [convert_to_tensor(x) for x in [position, rotation, velocity, omega]]
        if not device:
            [position, rotation, velocity, omega] = [x.to(device) for x in [position, rotation, velocity, omega]]

        # ((1+0.01*z)/(1+2z*0.01+0.01^2)) -> function of diff of (w,x,y,z) - (w,x,y,z+eps)
        rotation = F.normalize(rotation.unsqueeze(0)).squeeze(0)

        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.omega = omega
        self.object_name = object_name

    def toTensor(self):
        assert type(self.position) == torch.Tensor
        assert type(self.rotation) == torch.Tensor
        assert type(self.velocity) == torch.Tensor
        assert type(self.omega) == torch.Tensor
        assert self.object_name in REGISTERED_OBJECTS
        object_name_tensor = convert_obj_name_to_tensor(self.object_name)
        object_name_tensor = object_name_tensor.to(self.position.device)
        tensor = torch.cat([self.position, self.rotation, self.velocity, self.omega, object_name_tensor], dim=-1)
        assert tensor.shape[0] == EnvState.total_size
        return tensor

    @staticmethod
    def fromTensor(tensor):
        assert tensor.shape[0] == EnvState.total_size and len(tensor.shape) == 1
        position = tensor[0:3]
        rotation = tensor[3:7]
        velocity = tensor[7:10]
        omega = tensor[10:13]
        assert 0 <= tensor[EnvState.OBJECT_TYPE_INDEX] < len(REGISTERED_OBJECTS)
        object_name = convert_tensor_to_obj_name(tensor[EnvState.OBJECT_TYPE_INDEX])
        return EnvState(object_name, position, rotation, velocity, omega)

    def toTensorCoverName(self):
        assert type(self.position) == torch.Tensor
        assert type(self.rotation) == torch.Tensor
        assert type(self.velocity) == torch.Tensor
        assert type(self.omega) == torch.Tensor
        assert self.object_name in REGISTERED_OBJECTS
        object_name_tensor = torch.Tensor([-1.0]).float().to(self.position.device)
        tensor = torch.cat([self.position, self.rotation, self.velocity, self.omega, object_name_tensor], dim=-1)
        assert tensor.shape[0] == EnvState.total_size
        return tensor

    def clone(self):
        return EnvState(object_name=self.object_name, position=self.position.clone().detach(), rotation=self.rotation.clone().detach(), velocity=self.velocity.clone().detach(), omega=self.omega.clone().detach())

    def __str__(self):
        return 'object_name:{},position:{},rotation:{},velocity:{},omega:{}'.format(self.object_name, self.position, self.rotation, self.velocity, self.omega)

    def cuda_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cuda() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def cpu_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cpu() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def to_dict(self):
        return {'object_name': self.object_name, 'position': self.position.tolist(), 'rotation': self.rotation.tolist(),
                'velocity': self.velocity.tolist(), 'omega': self.omega.tolist()}


class NormEnvState:
    size = [3, 4, 3, 3]
    total_size = sum(size)

    def __init__(self, norm_or_denorm, position, rotation, position_mean, position_std, velocity_mean=None, velocity_std=None,
                 omega_mean=None, omega_std=None, velocity=None, omega=None, device=None):
        if velocity is None:
            self.use_vel = False
            velocity = torch.tensor([0., 0., 0.], device=position.device, requires_grad=False)
        else:
            self.use_vel = True
            velocity = norm_tensor(norm_or_denorm=norm_or_denorm, tensor=velocity, mean_tensor=velocity_mean, std_tensor=velocity_std)
        if omega is None:
            omega = torch.tensor([0., 0., 0.], device=position.device, requires_grad=False)
        else:
            omega = norm_tensor(norm_or_denorm=norm_or_denorm, tensor=omega, mean_tensor=omega_mean, std_tensor=omega_std)
        assert len(position) == 3 and len(rotation) == 4 and len(velocity) == 3 and len(omega) == 3
        position = norm_tensor(norm_or_denorm=norm_or_denorm, tensor=position, mean_tensor=position_mean, std_tensor=position_std)
        [position, rotation, velocity, omega] = [torch.Tensor(x) for x in [position, rotation, velocity, omega]]
        if not device:
            [position, rotation, velocity, omega] = [x.to(device) for x in [position, rotation, velocity, omega]]

        # ((1+0.01*z)/(1+2z*0.01+0.01^2)) -> function of diff of (w,x,y,z) - (w,x,y,z+eps)
        # rotation = F.normalize(rotation.unsqueeze(0)).squeeze(0)
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.omega = omega

    def toTensor(self):
        assert type(self.position) == torch.Tensor
        assert type(self.rotation) == torch.Tensor
        assert type(self.velocity) == torch.Tensor
        assert type(self.omega) == torch.Tensor
        if self.use_vel:
            tensor = torch.cat([self.position, self.rotation, self.velocity, self.omega], dim=-1)
            assert tensor.shape[0] == NormEnvState.total_size
        else:
            tensor = torch.cat([self.position, self.rotation], dim=-1)
        return tensor

    def __str__(self):
        return 'position:{},rotation:{},velocity:{},omega:{}'.format(self.position, self.rotation, self.velocity, self.omega)

    def cuda_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cuda() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def cpu_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cpu() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def to_dict(self):
        return {'position': self.position.tolist(), 'rotation': self.rotation.tolist(),
                'velocity': self.velocity.tolist(), 'omega': self.omega.tolist()}


class NpForceValOnly:
    size = [3]
    total_size = sum(size)

    def __init__(self, force):

        assert len(force) == 3

        force = np.array(force)

        self.force = force

    def __str__(self):
        return 'force:{}'.format(self.force)

    def tolist(self):
        return self.force.tolist()


class ForceValOnly:
    size = [3]
    total_size = sum(size)

    def __init__(self, force, device=None):

        assert len(force) == 3

        force = convert_to_tensor(force)
        if not device:
            force = force.to(device)

        self.force = force

    @staticmethod
    def fromTensor(tensor):
        assert len(tensor.shape) == 1 and tensor.shape[0] == ForceValOnly.total_size
        force = tensor
        return ForceValOnly(force)

    def __str__(self):
        return 'force:{}'.format(self.force)

    def cpu_(self):
        self.force = self.force.cpu()

    @staticmethod
    def fromForceArray(force_array):
        force_array_shape = force_array.shape
        if len(force_array_shape) == 2:
            return [ForceValOnly.fromTensor(force_array[cp_ind]) for cp_ind in range(force_array_shape[0])]
        elif len(force_array_shape) == 3:
            return [ForceValOnly.fromForceArray(force_array[seq_ind]) for seq_ind in range(force_array_shape[0])]
        else:
            raise Exception('Not implemented')

    def tolist(self):
        return self.force.cpu().tolist()

    def to(self, device):
        self.force = self.force.to(device)
        return self


def convert_to_tensor(x, require_grad=True):
    if type(x) == tuple or type(x) == list:
        result = torch.Tensor(x)
        if require_grad:
            result.requires_grad = True
        return result
    elif type(x) == torch.Tensor:
        return x
    else:
        import pdb;
        pdb.set_trace()
        raise Exception('Not implemented')


def convert_to_tuple(x):
    if type(x) == tuple:
        return x
    elif type(x) == torch.Tensor:
        x = x.cpu().tolist()
        return tuple(x)
    elif type(x) == np.ndarray:
        x = x.tolist()
        return tuple(x)
    else:
        import pdb;
        pdb.set_trace()
        raise Exception('Not implemented')
