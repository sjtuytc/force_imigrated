import torch


def norm_tensor(norm_or_denorm, tensor, mean_tensor, std_tensor):
    tensor, mean_tensor, std_tensor = torch.Tensor(tensor), torch.Tensor(mean_tensor), torch.Tensor(std_tensor)
    assert mean_tensor.shape == std_tensor.shape == tensor.shape, "tensor shape is not equal to its mean and std"
    if norm_or_denorm is None:
        return tensor
    elif norm_or_denorm:
        return (tensor - mean_tensor) / std_tensor
    else:
        return tensor * std_tensor + mean_tensor


def dict_of_tensor_to_cuda(tensor_dict):
    for feature in tensor_dict:
        value = tensor_dict[feature]
        if issubclass(type(value), torch.Tensor):
            tensor_dict[feature] = value.float().cuda(non_blocking=True)
    return tensor_dict

