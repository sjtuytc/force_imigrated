import torch
import glob
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


def clean_up_imgs(folder):
    print("Cleaning up images in folder!", folder)
    files = glob.glob(os.path.join(folder, '*.jpg'))
    for f in files:
        os.remove(f)


def generate_data(args, dataset, clean_dataset):
    # set up environment
    environment = MultipleObjectWrapper(
        environment=None, render=args.render, gravity=args.gravity, debug=args.debug,
        number_of_cp=args.number_of_cp, gpu_ids=args.gpu_ids, fps=args.fps,
        force_multiplier=args.force_multiplier, force_h=args.force_h,
        state_h=args.state_h, qualitative_size=args.qualitative_size, object_paths=dataset.object_paths)
    environment.reset()

    # set up folders
    debug_folder = 'debug/'
    root_dir = dataset.root_dir
    syn_dataset = os.path.join(root_dir, 'synthetic')
    os.makedirs(syn_dataset, exist_ok=True)

    for k, (input_data, target) in enumerate(dataset):
        obj_name, seq_img_paths = input_data['object_name'], input_data['image_paths']
        obj_folder = os.path.join(syn_dataset, obj_name)
        os.makedirs(obj_folder, exist_ok=True)

        env = environment.get_env_by_obj_name(object_name=obj_name)
        assert len(seq_img_paths) == len(target['rotation']) + 1 == len(target['position']) + 1, \
            "sequence length not match!"
        for idx, img_path in enumerate(seq_img_paths):
            if idx == 0:
                position, rotation = input_data['initial_position'], input_data['initial_rotation']
            else:
                position, rotation = target['position'][idx - 1], target['rotation'][idx - 1]
            env.update_object_transformations(object_state=NpEnvState(object_name=obj_name, position=position,
                                                                      rotation=rotation), object_num=None)
            folder, file_name = os.path.split(img_path)
            rendered_image = env.get_rgb()
            fp = save_image_to_disk(rendered_image.transpose(1, 0, 2), save_dir=obj_folder, save_name='syn_' + file_name)
        print(k, "/", len(dataset))


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
    generate_data(args=args, dataset=train_dataset, clean_dataset=True)

    test_dataset = get_dataset(args=args, train=False)
    logging.info("Generating synthetic dataset for test.")
    generate_data(args=args, dataset=test_dataset, clean_dataset=True)


if __name__ == "__main__":
    main()
