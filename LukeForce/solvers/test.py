import logging
import os
import time
import torch
from . import metrics
import tqdm
import numpy as np
from utils.visualization_util import vis_two_states, vis_state
from utils.tensor_utils import dict_of_tensor_to_cuda
from utils.data_io import save_into_pkl
from IPython import embed


def test_one_epoch(model, loss, data_loader, epoch, args):
    add_to_keys = 'Test'

    # Prepare model and optimizer
    model.eval()
    loss.eval()
    with torch.no_grad():
        # Setup average meters
        data_time_meter = metrics.AverageMeter()
        batch_time_meter = metrics.AverageMeter()
        metrics_time_meter = metrics.AverageMeter()
        backward_time_meter = metrics.AverageMeter()
        forward_pass_time_meter = metrics.AverageMeter()
        loss_time_meter = metrics.AverageMeter()
        loss_meter = metrics.AverageMeter()
        accuracy_metric = [m(args) for m in model.metric]
        loss_detail_meter = {loss_name: metrics.AverageMeter() for loss_name in loss.local_loss_dict}

        # Iterate over data
        timestamp = time.time()
        all_results = []
        for i, (input_dict, target_dict) in enumerate(tqdm.tqdm(data_loader)):
            # Move data to gpu
            if 'rgb' in input_dict.keys():
                batch_size = input_dict['rgb'].size(0)
            else:
                batch_size = input_dict['norm_force'].size(0)

            if args.gpu_ids != -1:
                input_dict = dict_of_tensor_to_cuda(input_dict)
                target_dict = dict_of_tensor_to_cuda(target_dict)
                if 'statistics' in target_dict.keys():
                    target_dict['statistics'] = dict_of_tensor_to_cuda(target_dict['statistics'])
            data_time_meter.update((time.time() - timestamp) / batch_size, batch_size)

            before_forward_pass_time = time.time()
            # Forward pass
            model_output, target_output = model(input_dict, target_dict)
            target_output['loss1_or_loss2'] = None
            # if args.save_dataset:  # deprecating.
            #     # input_dict: ['rgb', 'initial_position', 'initial_rotation', 'initial_keypoint', 'object_name',
            #     # 'contact_points', 'timestamps']
            #     # target_dict: ['keypoints', 'position', 'rotation', 'contact_points', 'object_name']
            #     # model_output: ['keypoints', 'rotation', 'position', 'force_success_flag', 'force_applied',
            #     # 'force_direction', 'contact_points']
            #     cur_result = {'initial_position': input_dict['initial_position'].cpu().tolist()[0],
            #                   'initial_rotation': input_dict['initial_rotation'].cpu().tolist()[0],
            #                   'object_name': input_dict['object_name'][0],
            #                   'model_contact_points': model_output['contact_points'].cpu().tolist()[0],
            #                   'timestamps': input_dict['timestamps'], 'model_position': model_output['position'].cpu().tolist()[0],
            #                   'model_rotation': model_output['rotation'].cpu().tolist()[0],
            #                   'force_applied': model_output['force_applied'].cpu().tolist()[0]}
            #     all_results.append(cur_result)
            #     if i % args.save_freq == 0 or i == len(data_loader) - 1:
            #         save_into_pkl(all_results, folder=args.ns_dataset_p, name='all_data', verbose=True)
            forward_pass_time_meter.update((time.time() - before_forward_pass_time) / batch_size, batch_size)

            before_loss_time = time.time()
            loss_output = loss(model_output, target_output)
            vis_cur_iter = args.vis and i < 20
            if vis_cur_iter:
                init_pos, init_rot, obj_name, init_image = input_dict['initial_position'][0].cpu(), input_dict['initial_rotation'][0].cpu(), \
                                                           target_output['object_name'][0], input_dict['image_paths'][0][0]
                vis_state(vis_env=args.vis_env, obj_name=obj_name, position=init_pos, rotation=init_rot, full_image=init_image,
                          image_name=args.title + '_init_' + str(i), save_folder=args.vis_f, set_color=3, verbose=True)

                def vis_fn(seq_id):
                    phy_pos, phy_rot = model_output['phy_position'][0][seq_id].cpu(), model_output['phy_rotation'][0][seq_id].cpu()
                    ns_pos, ns_rot = model_output['ns_position'][0][seq_id].cpu(), model_output['ns_rotation'][0][seq_id].cpu()
                    gt_pos, gt_rot = target_output['position'][0][seq_id].cpu(), target_output['rotation'][0][seq_id].cpu()
                    full_image_p = input_dict['image_paths'][seq_id + 1][0]
                    ns_color, phy_color, gt_color = 0, 1, 2  # green, blue, orange
                    vis_two_states(vis_env=args.vis_env, obj_name=obj_name, pos1=phy_pos, rot1=phy_rot, pos2=gt_pos, rot2=gt_rot,
                                   full_image=full_image_p, image_name=args.title + '_phy_gt_' + str(i) + '_' + str(seq_id),
                                   save_folder=args.vis_f, color1=phy_color, color2=gt_color,
                                   kp_tensor=model.all_objects_keypoint_tensor[obj_name], verbose=True)
                    vis_two_states(vis_env=args.vis_env, obj_name=obj_name, pos1=ns_pos, rot1=ns_rot, pos2=gt_pos, rot2=gt_rot,
                                   full_image=full_image_p, image_name=args.title + '_ns_gt_' + str(i) + '_' + str(seq_id),
                                   save_folder=args.vis_f, color1=ns_color, color2=gt_color,
                                   kp_tensor=model.all_objects_keypoint_tensor[obj_name], verbose=True)
                    vis_two_states(vis_env=args.vis_env, obj_name=obj_name, pos1=phy_pos, rot1=phy_rot, pos2=ns_pos, rot2=ns_rot,
                                   full_image=full_image_p, image_name=args.title + '_phy_ns_' + str(i) + '_' + str(seq_id),
                                   save_folder=args.vis_f, color1=phy_color, color2=ns_color,
                                   kp_tensor=model.all_objects_keypoint_tensor[obj_name], verbose=True)
                vis_fn(1)
                vis_fn(2)
                vis_fn(3)
            loss_time_meter.update((time.time() - before_loss_time) / batch_size, batch_size)
            if args.render:
                print('loss', loss.local_loss_dict)

            before_backward_time = time.time()
            backward_time_meter.update((time.time() - before_backward_time) / batch_size, batch_size)

            model_output = {f: model_output[f].detach() for f in model_output.keys()}

            # Bookkeeping on loss, accuracy, and batch time
            loss_meter.update(loss_output.detach(), batch_size)
            before_metrics_time = time.time()
            with torch.no_grad():
                for acc in accuracy_metric:
                    acc.record_output(model_output, target_output)
            metrics_time_meter.update((time.time() - before_metrics_time) / batch_size, batch_size)
            batch_time_meter.update((time.time() - timestamp), batch_size)

            # Log report
            dataset_length = len(data_loader.dataset)
            real_index = (epoch - 1) * dataset_length + (i * args.batch_size)

            loss_values = loss.local_loss_dict
            for loss_name in loss_detail_meter:
                if loss_values[loss_name] is None:
                    continue
                (loss_val, data_size) = loss_values[loss_name]
                loss_detail_meter[loss_name].update(loss_val.item(), data_size)

            if i % args.tensorboard_log_freq == 0:
                result_log_dict = {
                    'Time/Batch': batch_time_meter.avg,
                    'Time/Data': data_time_meter.avg,
                    'Time/Metrics': metrics_time_meter.avg,
                    'Time/backward': backward_time_meter.avg,
                    'Time/forward': forward_pass_time_meter.avg,
                    'Time/loss': loss_time_meter.avg,
                    'Loss': loss_meter.avg,
                }

                for loss_name in loss_detail_meter:
                    result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

                for ac in accuracy_metric:
                    result_log_dict[type(ac).__name__] = ac.average()
                args.logging_module.log(result_log_dict, real_index + 1, add_to_keys=add_to_keys)

            timestamp = time.time()

            result_log_dict = {
                'Time/Batch': batch_time_meter.avg,
                'Time/Data': data_time_meter.avg,
                'Time/Metrics': metrics_time_meter.avg,
                'Time/backward': backward_time_meter.avg,
                'Time/forward': forward_pass_time_meter.avg,
                'Time/loss': loss_time_meter.avg,
                'Loss': loss_meter.avg,
            }

            for loss_name in loss_detail_meter:
                result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

            with torch.no_grad():
                for ac in accuracy_metric:
                    result_log_dict[type(ac).__name__] = ac.average()
            args.logging_module.log(result_log_dict, epoch, add_to_keys=add_to_keys + '_Summary')

            testing_summary = ('Epoch: [{}] -- TESTING SUMMARY\t'.format(epoch) +
                               'Time {batch_time.sum:.2f};   Data {data_time.sum:.2f};  Loss {loss.avg:.6f};   '
                               '{accuracy_report}'.format(batch_time=batch_time_meter, data_time=data_time_meter,
                                                          loss=loss_meter,
                                                          accuracy_report='\n'.join([ac.final_report()
                                                                                     for ac in accuracy_metric])))

            if (i % 50 == 0 and args.mode == 'test') or i == len(data_loader) - 1 or vis_cur_iter:
                logging.info(testing_summary)
                logging.info('Full test result is at {}'.format(
                    os.path.join(args.save, 'test.log')))
                if i == len(data_loader) - 1:  # only save at the end of this epoch
                    with open(os.path.join(args.save, 'test.log'), 'a') as fp:
                        fp.write('{}\n'.format(testing_summary))
    return loss_meter.avg
