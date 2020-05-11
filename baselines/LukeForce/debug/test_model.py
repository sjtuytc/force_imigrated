import torch
import logging
import random
import os
import time
import matplotlib as mpl
import random
random.seed(0)
from solvers.metrics import AverageMeter

mpl.use('Agg')

from utils.arg_parser import parse_args
from utils.visualization_util import *
from environments.env_wrapper_multiple_object import MultipleObjectWrapper
from utils.environment_util import NpEnvState


def train_one_data(model, loss, optimizer, test_dataset, args):

    # Prepare model and optimizer
    model.train()
    loss.train()
    lr = model.learning_rate(1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    accuracy_metric = [m(args) for m in model.metric]
    loss_detail_meter = {loss_name: AverageMeter() for loss_name in loss.local_loss_dict}

    input_data, target = test_dataset[0]
    # ----input data shapes----
    # 'rgb' tensor: [seq len, 3, 224, 224]; 'initial_position' tensor: [3];
    # 'initial_rotation' tensor: [4]; 'initial_keypoint' tensor: [10, 2];
    # 'object_name' list: [1], 'contact_points' tensor: [5, 3], 'timestamps' list: [10].
    # ----target data shapes----
    # 'keypoints': [9, 10, 2], 'contact_points': [5, 3], 'position': [9, 3], 'rotation': [9, 4]

    # visualize input data and target.
    # loaded_image = load_image_from_disk(full_path=input_data['image_paths'][0])
    # full_path = save_image_to_disk(image_array=loaded_image, save_name='input_rgb', save_dir=args.save)
    # print('Input image saved at ', full_path, ".")

    environment = MultipleObjectWrapper(
        environment=None, render=args.render, gravity=args.gravity, debug=args.debug,
        number_of_cp=args.number_of_cp, gpu_ids=args.gpu_ids, fps=args.fps,
        force_multiplier=args.force_multiplier, force_h=args.force_h,
        state_h=args.state_h, qualitative_size=args.qualitative_size, object_paths=test_dataset.object_paths)
    environment.reset()
    initial_env_state = NpEnvState(object_name=input_data['object_name'], position=input_data['initial_position'],
                                     rotation=input_data['initial_rotation'])
    vis_env = environment.get_env_by_obj_name(object_name=input_data['object_name'])
    simu_rgb = vis_env.get_rgb_for_position_rotation(object_state=initial_env_state,
                                                     contact_point=input_data['contact_points'])
    simu_path = save_image_to_disk(image_array=simu_rgb, save_name='simu_rgb', save_dir=args.save)
    print("Env image saved at ", simu_path, ".")

    # add bs in dims.
    batch_size = 1
    input_data['rgb'], input_data['initial_position'], input_data['initial_rotation'] = \
        input_data['rgb'].unsqueeze(0), input_data['initial_position'].unsqueeze(0), input_data['initial_rotation'].unsqueeze(0)
    input_data['initial_keypoint'], input_data['object_name'], input_data['contact_points'], input_data['timestamps'] = \
        input_data['initial_keypoint'].unsqueeze(0), [input_data['object_name']], \
        input_data['contact_points'].unsqueeze(0), \
        [input_data['timestamps']]
    target['keypoints'], target['position'], target['rotation'], target['contact_points'] = \
        target['keypoints'].unsqueeze(0), target['position'].unsqueeze(0), target['rotation'].unsqueeze(0), \
        target['contact_points'].unsqueeze(0)

    if args.gpu_ids != -1:  # if use gpu
        for feature in input_data:
            value = input_data[feature]
            if issubclass(type(value), torch.Tensor):
                input_data[feature] = value.cuda(non_blocking=True)
            target = {feature: target[feature].cuda(non_blocking=True) for feature in target.keys()}


    # Forward pass
    model_output, target_output = model(input_data, target)
    loss_output = loss(model_output, target_output)
    loss_output.backward()
    model_output = {f: model_output[f].detach() for f in model_output.keys()}

    # test optimizer
    optimizer.step()
    optimizer.zero_grad()

    # visualization results
    # get_image_list_for_viz_cp(output_cp=model_output['contact_points'], target_cp=target_output['contact_points'],
    #                           environment=None)

    # Bookkeeping on loss, accuracy, and batch time
    loss_meter.update(loss_output.detach(), batch_size)
    with torch.no_grad():
        for acc in accuracy_metric:
            acc.record_output(model_output, target_output)

    loss_values = loss.local_loss_dict
    for loss_name in loss_detail_meter:
        (loss_val, data_size) = loss_values[loss_name]
        loss_detail_meter[loss_name].update(loss_val.item(), data_size)

    if batch_time_meter is not None and data_time_meter is not None and loss_meter is not None:
        training_summary = ('Epoch: [{}] -- TRAINING SUMMARY\t'.format(0) +
                            'Time {batch_time:.2f}   Data {data_time:.2f}  Loss {loss:.6f}  {accuracy_report}'.
                            format(batch_time=batch_time_meter.avg, data_time=data_time_meter.avg, loss=loss_meter.avg,
                                   accuracy_report='\n'.join([ac.final_report() for ac in accuracy_metric])))
    else:
        training_summary = ""
    print(training_summary)


def get_dataset(args):
    print("Create training dataset.")
    debug_dataset = args.dataset(args, train=True)
    return debug_dataset


def get_model_and_loss(args):
    model = args.model(args)
    restarting_epoch = 0
    if args.gpu_ids != -1:
        model = model.cuda()
    loss = model.loss(args)
    if args.gpu_ids != -1:
        loss = loss.cuda()
    return model, loss, restarting_epoch


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Reading dataset metadata.')
    debug_dataset = get_dataset(args)

    logging.info('Constructing model.')
    model, loss, restarting_epoch = get_model_and_loss(args)
    print("Debug model construction finished!")

    optimizer = model.optimizer()
    train_one_data(model=model, loss=loss, optimizer=optimizer, test_dataset=debug_dataset, args=args)


if __name__ == "__main__":
    main()
