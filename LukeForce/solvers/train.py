import logging
import os
import time
import torch
from . import metrics
import tqdm
from utils.tensor_utils import dict_of_tensor_to_cuda


def train_one_epoch(model, loss, optimizer, data_loader, epoch, args):
    vis_grad = args.vis_grad
    add_to_keys = 'Train'

    # Prepare model and optimizer
    model.train()
    loss.train()
    lr = model.learning_rate(epoch)
    ns_model = optimizer is None  # using neural force simulator
    if ns_model:
        model.set_learning_rate(lr=lr)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Setup average meters
    data_time_meter = metrics.AverageMeter()
    batch_time_meter = metrics.AverageMeter()
    metrics_time_meter = metrics.AverageMeter()
    backward_time_meter = metrics.AverageMeter()
    forward_pass_time_meter = metrics.AverageMeter()
    loss_time_meter = metrics.AverageMeter()
    loss_meter = metrics.AverageMeter()
    loss1_grad_meter = metrics.AverageMeter()
    loss2_grad_meter = metrics.AverageMeter()
    loss_cp_grad_meter = metrics.AverageMeter()
    accuracy_metric = [m(args) for m in model.metric]
    loss_detail_meter = {loss_name: metrics.AverageMeter() for loss_name in loss.local_loss_dict}

    # Iterate over data
    timestamp = time.time()
    print("Begin train one epoch!")
    loss1_or_loss2 = True  # if true, update loss1; else, update loss2. If None, learn both.
    for i, (input_dict, target_dict) in enumerate(tqdm.tqdm(data_loader)):
        if 'rgb' in input_dict.keys():
            batch_size = input_dict['rgb'].size(0)
        else:
            batch_size = input_dict['norm_force'].size(0)
        if args.gpu_ids != -1:  # if use gpu
            input_dict = dict_of_tensor_to_cuda(input_dict)
            target_dict = dict_of_tensor_to_cuda(target_dict)
            if 'statistics' in target_dict:
                target_dict['statistics'] = dict_of_tensor_to_cuda(target_dict['statistics'])
            data_time_meter.update((time.time() - timestamp) / batch_size, batch_size)

            before_forward_pass_time = time.time()
            # Forward pass
            model_output, target_output = model(input_dict, target_dict)
            target_output['loss1_or_loss2'] = loss1_or_loss2
            forward_pass_time_meter.update((time.time() - before_forward_pass_time) / batch_size, batch_size)
            before_loss_time = time.time()
            loss_output = loss(model_output, target_output)
            loss_time_meter.update((time.time() - before_loss_time) / batch_size, batch_size)

            before_backward_time = time.time()
            loss_output.backward()
            backward_time_meter.update((time.time() - before_backward_time) / batch_size, batch_size)

            model_output = {f: model_output[f].detach() for f in model_output.keys()}
            if i % args.break_batch == 0 or i == len(data_loader) - 1:
                if optimizer is None:
                    model.step_optimizer(loss1_or_loss2)
                    loss1_or_loss2 = not loss1_or_loss2   # alternatively train two targets.
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                if vis_grad:  # check grads and then clear them
                    cp_grad, loss1_grad, loss2_grad = loss.seperate_loss_backward(
                        input_dict=input_dict, target_dict=target_dict, optimizer=model.get_optim(), model_obj=model)
                    loss_cp_grad_meter.update(cp_grad, 1)
                    loss1_grad_meter.update(loss1_grad, 1)
                    loss2_grad_meter.update(loss2_grad, 1)
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
                if vis_grad:
                    result_log_dict['Grad/loss1'] = loss1_grad_meter.avg
                    result_log_dict['Grad/loss2'] = loss2_grad_meter.avg
                    result_log_dict['Grad/loss_cp'] = loss_cp_grad_meter.avg
                for loss_name in loss_detail_meter:
                    result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

                for ac in accuracy_metric:
                    result_log_dict[type(ac).__name__] = ac.average()  # record average for every object.
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

        if batch_time_meter is not None and data_time_meter is not None and loss_meter is not None:
            training_summary = ('Epoch: [{}] -- TRAINING SUMMARY\t'.format(epoch) +
                                'Time {batch_time:.2f}   Data {data_time:.2f}  Loss {loss:.6f}  {accuracy_report}'.
                                format(batch_time=batch_time_meter.avg, data_time=data_time_meter.avg, loss=loss_meter.avg,
                                       accuracy_report='\n'.join([ac.final_report() for ac in accuracy_metric])))
        else:
            training_summary = ""
        if i % 50 == 0 or i == len(data_loader) - 1:
            logging.info(training_summary)
            logging.info('Full train result is at {}'.format(os.path.join(args.save, 'train.log')))
            if i == len(data_loader) - 1:  # only save at the end of this epoch
                with open(os.path.join(args.save, 'train.log'), 'a') as fp:
                    fp.write('{}\n'.format(training_summary))
    return loss_meter.avg
