import torch
import glob
import cv2
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
from utils.data_io import save_into_pkl, save_into_json
from utils.projection_utils import *
from utils.grounding_util import get_bbox_from_keypoints


def generate_data(args, dataset, clean_dataset, save_prefix):
    # set up environment
    environment = MultipleObjectWrapper(
        environment=None, render=args.render, gravity=args.gravity, debug=args.debug,
        number_of_cp=args.number_of_cp, gpu_ids=args.gpu_ids, fps=args.fps,
        force_multiplier=args.force_multiplier, force_h=args.force_h,
        state_h=args.state_h, qualitative_size=args.qualitative_size, object_paths=dataset.object_paths)
    environment.reset()

    # set up folders
    final_dict = {}
    debug_folder = 'debug/'
    root_dir = dataset.root_dir
    syn_dataset = os.path.join(root_dir, 'synthetic')
    os.makedirs(syn_dataset, exist_ok=True)
    obj_name_to_vertex, obj_name_to_center_ob_mass = {}, {}
    for obj_name in environment.list_of_envs.keys():
        env = environment.get_env_by_obj_name(object_name=obj_name)
        obj_name_to_vertex[obj_name] = env.vertex_points
        obj_name_to_center_ob_mass[obj_name] = env.center_of_mass
    for k, (input_data, target) in enumerate(dataset):
        obj_name, seq_img_paths, time_seqs = input_data['object_name'], input_data['image_paths'], \
                                             input_data['timestamps']
        obj_folder = os.path.join(syn_dataset, obj_name)
        os.makedirs(obj_folder, exist_ok=True)
        if obj_name not in final_dict.keys():
            final_dict[obj_name] = {}

        assert len(seq_img_paths) == len(time_seqs) == len(target['rotation']) + 1 == len(target['position']) + 1, \
            "sequence length not match!"
        for idx, img_path in enumerate(seq_img_paths):
            time_label = time_seqs[idx]
            if idx == 0:
                position, rotation = input_data['initial_position'], input_data['initial_rotation']
            else:
                position, rotation = target['position'][idx - 1], target['rotation'][idx - 1]
            projected_all = np_get_model_vertex_projection(set_of_points=obj_name_to_vertex[obj_name],
                                                           center_of_mass=obj_name_to_center_ob_mass[obj_name],
                                                           object_name=input_data['object_name'],
                                                           position=np.array(position),
                                                           rotation=np.array(rotation))

            x1, y1, x2, y2 = get_bbox_from_keypoints(projected_kp=projected_all, img_w=1920, img_h=1080, scale_ratio=1.5)
            result = {'bbox': [x1, y1, x2, y2], 'project_points': projected_all.tolist()}
            if time_label in final_dict[obj_name].keys():
                assert result['bbox'] == final_dict[obj_name][time_label]['bbox']
            final_dict[obj_name][time_label] = result

            # visualize annotations.
            # loaded_image = load_image_from_disk(full_path=img_path)
            # image_with_all = put_keypoints_on_image(image=loaded_image, keypoints=projected_all, coloring=True,
            #                                         SIZE_OF_DOT=5, exchange_x_y=True)
            # simu_path = save_image_to_disk(image_array=image_with_all, save_name=str(time_label) + 'vis_projected_all',
            #                                save_dir=debug_folder)
            # image_with_all = cv2.imread(simu_path)
            # image_with_bbox = cv2.rectangle(image_with_all, (x1, y1), (x2, y2), (0, 255, 0), 5)
            # cv2.imwrite(os.path.join(debug_folder, str(time_label) + 'img_with_bbox.jpg'), image_with_bbox)
        print("Generating:", k, "/", len(dataset))
        if k % 100 == 0 or k == len(dataset) - 1:
            save_folder = os.path.join(dataset.root_dir, 'annotations')
            save_into_json(final_dict, folder=save_folder, file_name=save_prefix + "_syn_time_to_bbox")


def get_dataset(args, train=True):
    print("Create training dataset.")
    # don't use environment.
    debug_dataset = args.dataset(args, environment=-1, train=train)
    return debug_dataset


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = get_dataset(args, train=True)
    logging.info("Generating synthetic dataset for train.")
    generate_data(args=args, dataset=train_dataset, clean_dataset=True, save_prefix="train")

    test_dataset = get_dataset(args=args, train=False)
    logging.info("Generating synthetic dataset for test.")
    generate_data(args=args, dataset=test_dataset, clean_dataset=True, save_prefix="test")


if __name__ == "__main__":
    main()
