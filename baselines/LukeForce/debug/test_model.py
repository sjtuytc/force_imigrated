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
from utils.data_io import save_into_pkl
from utils.projection_utils import *


def train_one_data(model, loss, optimizer, test_dataset, args):
    debug_folder = 'debug/'
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

    data_idx = [0, 1000, 2000, 3000, 4000, 5000]
    for one_idx in data_idx:
        input_data, target = test_dataset[one_idx]
        # ----input data shapes----
        # 'rgb' tensor: [seq len, 3, 224, 224]; 'initial_position' tensor: [3];
        # 'initial_rotation' tensor: [4]; 'initial_keypoint' tensor: [10, 2];
        # 'object_name' list: [1], 'contact_points' tensor: [5, 3], 'timestamps' list: [10].
        # ----target data shapes----
        # 'keypoints': [9, 10, 2], 'contact_points': [5, 3], 'position': [9, 3], 'rotation': [9, 4]

        # visualize input data and target.
        loaded_image = load_image_from_disk(full_path=input_data['image_paths'][0])

        # full_path = save_image_to_disk(image_array=loaded_image, save_name='input_rgb', save_dir=debug_folder)
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

        # visualize 3D model points
        vis_3d = {'mp': np.array(vis_env.vertex_points), 'cp': np.array(input_data['contact_points']),
                  'kp': np.array(model.all_objects_keypoint_tensor[input_data['object_name']]),
                  'name': input_data['object_name']}
        save_into_pkl(vis_3d, folder=debug_folder, name=str(one_idx) + "_vis_3d_metadata", verbose=True)

        # visualize projected model points
        projected_all = np_get_model_vertex_projection(set_of_points=vis_env.vertex_points, center_of_mass=vis_env.center_of_mass,
                                                       object_name=input_data['object_name'],
                                                       position=np.array(input_data['initial_position']),
                                                       rotation=input_data['initial_rotation'])
        image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=projected_all, coloring=True, SIZE_OF_DOT=5,
                                                exchange_x_y=True)
        simu_path = save_image_to_disk(image_array=image_with_all, save_name=str(one_idx) + '_projected_model', save_dir=debug_folder)
        print("Image with all model points saved at ", simu_path, ".")

        # visualize gt keypoints
        image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=np.array(input_data['initial_keypoint']),
                                                coloring=True, SIZE_OF_DOT=5, exchange_x_y=True)
        save_p = save_image_to_disk(image_array=image_with_all, save_name=str(one_idx) + '_gt_kp', save_dir=debug_folder)
        print("Image with gt keypoints saved at ", save_p, ".")

        # visualize projected keypoints
        projected_kp = np_get_keypoint_projection(object_name=input_data['object_name'], position=input_data['initial_position'],
                                                  rotation=input_data['initial_rotation'],
                                                  keypoints=model.all_objects_keypoint_tensor[input_data['object_name']])
        image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=projected_kp, coloring=True, SIZE_OF_DOT=5,
                                                exchange_x_y=True)
        simu_path = save_image_to_disk(image_array=image_with_all, save_name=str(one_idx) + '_projected_kp', save_dir=debug_folder)
        print("Image with projected keypoints saved at ", simu_path, ".")

        # visualize projected contact points
        projected_cp = np_get_cp_projection(set_of_points=input_data['contact_points'], center_of_mass=None,
                                            object_name=input_data['object_name'], position=input_data['initial_position'],
                                            rotation=input_data['initial_rotation'])
        image_with_cp = put_keypoints_on_image(image=loaded_image, keypoints=projected_cp, coloring=True, SIZE_OF_DOT=5,
                                               exchange_x_y=True)
        simu_path = save_image_to_disk(image_array=image_with_cp, save_name=str(one_idx) + '_projected_cp', save_dir=debug_folder)
        print("Image with projected CP saved at ", simu_path, ".")

    raise RuntimeError

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
