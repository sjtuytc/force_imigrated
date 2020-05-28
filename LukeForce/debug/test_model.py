import torch
import logging
import random
import os
import time
import matplotlib as mpl
import random
random.seed(0)
from solvers.metrics import AverageMeter
import shutil

mpl.use('Agg')

from utils.arg_parser import parse_args
from utils.visualization_util import *
from environments.env_wrapper_multiple_object import MultipleObjectWrapper
from utils.environment_util import NpEnvState
from utils.data_io import save_into_pkl
from utils.projection_utils import *


def visualize_input_data(dataset, visualize_ind, environment):
    input_data, target = dataset[visualize_ind]
    # ----input data shapes----
    # 'rgb' tensor: [seq len, 3, 224, 224]; 'initial_position' tensor: [3];
    # 'initial_rotation' tensor: [4]; 'initial_keypoint' tensor: [10, 2];
    # 'object_name' list: [1], 'contact_points' tensor: [5, 3], 'timestamps' list: [10].
    # ----target data shapes----
    # 'keypoints': [9, 10, 2], 'contact_points': [5, 3], 'position': [9, 3], 'rotation': [9, 4]

    # visualize input data and target.
    loaded_image = load_image_from_disk(full_path=input_data['image_paths'][0])
    print("Load original image from ", input_data['image_paths'][0])
    # full_path = save_image_to_disk(image_array=loaded_image, save_name='input_rgb', save_dir=debug_folder)
    # print('Input image saved at ', full_path, ".")

    vis_env = environment.get_env_by_obj_name(object_name=input_data['object_name'])

    # # visualize 3D model points
    # vis_3d = {'mp': np.array(vis_env.vertex_points), 'cp': np.array(input_data['contact_points']),
    #           'kp': np.array(model.all_objects_keypoint_tensor[input_data['object_name']]),
    #           'name': input_data['object_name']}
    # save_into_pkl(vis_3d, folder=debug_folder, name=str(one_idx) + "_vis_3d_metadata", verbose=True)

    # # visualize projected model points
    projected_all = np_get_model_vertex_projection(set_of_points=vis_env.vertex_points,
                                                   center_of_mass=vis_env.center_of_mass,
                                                   object_name=input_data['object_name'],
                                                   position=np.array(input_data['initial_position']),
                                                   rotation=input_data['initial_rotation'])
    image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=projected_all, coloring=True,
                                            SIZE_OF_DOT=5, exchange_x_y=True)
    # simu_path = save_image_to_disk(image_array=image_with_all, save_name=str(one_idx) + '_projected_model', save_dir=debug_folder)
    # print("Image with all model points saved at ", simu_path, ".")

    # visualize gt keypoints
    image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=np.array(input_data['initial_keypoint']),
                                            coloring=True, SIZE_OF_DOT=5, exchange_x_y=True)
    save_p = save_image_to_disk(image_array=image_with_all, save_name=str(one_idx) + '_gt_kp', save_dir=debug_folder)
    print("Image with gt keypoints saved at ", save_p, ".")

    # # visualize projected keypoints
    # projected_kp = np_get_keypoint_projection(object_name=input_data['object_name'], position=input_data['initial_position'],
    #                                           rotation=input_data['initial_rotation'],
    #                                           keypoints=model.all_objects_keypoint_tensor[input_data['object_name']])
    # image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=projected_kp, coloring=True, SIZE_OF_DOT=5,
    #                                         exchange_x_y=True)
    # simu_path = save_image_to_disk(image_array=image_with_all, save_name=str(one_idx) + '_projected_kp', save_dir=debug_folder)
    # print("Image with projected keypoints saved at ", simu_path, ".")
    #
    # # visualize projected contact points
    # projected_cp = np_get_cp_projection(set_of_points=input_data['contact_points'], center_of_mass=None,
    #                                     object_name=input_data['object_name'], position=input_data['initial_position'],
    #                                     rotation=input_data['initial_rotation'])
    # image_with_cp = put_keypoints_on_image(image=loaded_image, keypoints=projected_cp, coloring=True, SIZE_OF_DOT=5,
    #                                        exchange_x_y=True)
    # simu_path = save_image_to_disk(image_array=image_with_cp, save_name=str(one_idx) + '_projected_cp', save_dir=debug_folder)
    # print("Image with projected CP saved at ", simu_path, ".")

    return


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

    environment = MultipleObjectWrapper(
        environment=None, render=args.render, gravity=args.gravity, debug=args.debug,
        number_of_cp=args.number_of_cp, gpu_ids=args.gpu_ids, fps=args.fps,
        force_multiplier=args.force_multiplier, force_h=args.force_h,
        state_h=args.state_h, qualitative_size=args.qualitative_size, object_paths=test_dataset.object_paths)
    environment.reset()

    # data_idx = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
    # for one_idx in data_idx:
    #     visualize_input_data(test_dataset, one_idx, environment)

    # add bs in dims
    input_data, target = test_dataset[0]
    batch_size = 1
    input_data['rgb'], input_data['initial_position'], input_data['initial_rotation'] = \
        input_data['rgb'].unsqueeze(0), input_data['initial_position'].unsqueeze(0), input_data['initial_rotation'].unsqueeze(0)
    input_data['initial_keypoint'], input_data['object_name'], input_data['contact_points'], input_data['timestamps'] = \
        input_data['initial_keypoint'].unsqueeze(0), [input_data['object_name']], \
        input_data['contact_points'].unsqueeze(0), \
        [input_data['timestamps']]
    input_data['syn_rgb'] = input_data['syn_rgb'].unsqueeze(0)
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

    print(model_output['keypoints'].shape, target_output['keypoints'].shape)

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
    train_dataset = args.dataset(args, environment=None, train=True, bbox_gt=True, scale=512)
    test_dataset = args.dataset(args, environment=None, train=False, bbox_gt=True, scale=512)
    return train_dataset, test_dataset


def get_model_and_loss(args):
    model = args.model(args)
    restarting_epoch = 0
    if args.gpu_ids != -1:
        model = model.cuda()
    if args.reload:
        reload_adr = args.reload
    else:
        reload_adr = None
    if reload_adr is not None:
        if args.gpu_ids == -1:
            loaded_weights = torch.load(reload_adr, map_location='cpu')
        else:
            loaded_weights = torch.load(reload_adr)
        model.load_state_dict(loaded_weights, strict=args.strict)
    loss = model.loss(args)
    if args.gpu_ids != -1:
        loss = loss.cuda()
    return model, loss, restarting_epoch


def save_image_to_dir(obj2image_dict, time2image_path, target_folder, obj_name="019_pitcher_base"):
    time_seq_cur_obj = obj2image_dict[obj_name]
    time2image_cur_obj = time2image_path[obj_name]
    for one_time in time_seq_cur_obj:
        one_image_p = time2image_cur_obj[one_time]['image_adr']
        one_image_p = one_image_p.replace('LMJTFY/', '')
        one_image_p = os.path.join('DatasetForce', one_image_p)
        shutil.copyfile(one_image_p, os.path.join(target_folder, one_time + ".jpg"))
        print("copy from ", one_image_p, "to", target_folder)
    return


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Reading dataset metadata.')
    train_dataset, test_dataset = get_dataset(args)
    train_obj2image, test_obj2image = train_dataset.final_obj2image, test_dataset.final_obj2image
    train_demo, test_demo = 'demo_data/train', 'demo_data/test'
    os.makedirs(train_demo, exist_ok=True)
    os.makedirs(test_demo, exist_ok=True)
    save_image_to_dir(train_obj2image, train_dataset.time_to_clip_ind_image_adr, train_demo)
    save_image_to_dir(test_obj2image, test_dataset.time_to_clip_ind_image_adr, test_demo)
    raise RuntimeError("finish making demo")
    logging.info('Constructing model.')
    model, loss, restarting_epoch = get_model_and_loss(args)
    print("Debug model construction finished!")

    optimizer = model.optimizer()
    train_one_data(model=model, loss=loss, optimizer=optimizer, test_dataset=train_dataset, args=args)


if __name__ == "__main__":
    main()
