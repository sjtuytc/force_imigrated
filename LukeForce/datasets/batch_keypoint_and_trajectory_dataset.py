import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pdb
from environments.subproc_physics_env import SubprocPhysicsEnv
from utils.data_loading_utils import load_json_dict, get_time_from_str, scale_position, process_projection


def calculate_data_stats(final_data):
    obj_name2images = {}
    for idx, one_d in enumerate(final_data):
        obj_name, sequence = one_d['obj_name'], one_d['sequence']
        if obj_name not in obj_name2images.keys():
            obj_name2images[obj_name] = []
        for one_time in sequence:
            if one_time not in obj_name2images[obj_name]:
                obj_name2images[obj_name].append(one_time)
    return obj_name2images


class BatchDatasetWAugmentation(data.Dataset):
    def __init__(self, args, environment, bbox_gt=True, scale=224, train=True):
        # in original paper, scale is 224
        self.root_dir = args.data
        self.object_list = args.object_list
        self.number_of_cp = args.number_of_cp
        self.fps = args.fps
        if train:
            name_of_time_to_clip = 'train_time_to_clip_ind.json'
        elif args.use_val:
            name_of_time_to_clip = 'val_time_to_clip_ind.json'
        else:
            name_of_time_to_clip = 'test_time_to_clip_ind.json'

        time_to_clip_ind_image_adr_json = os.path.join(self.root_dir, 'annotations', name_of_time_to_clip)
        self.time_to_clip_ind_image_adr = load_json_dict(time_to_clip_ind_image_adr_json)

        clip_to_contact_point_json_file = os.path.join(self.root_dir, 'annotations', 'clean_clip_to_contact_point.json')
        self.clip_to_contact_point = load_json_dict(clip_to_contact_point_json_file)

        time_to_obj_state_fps_json_file = os.path.join(self.root_dir, 'annotations', 'time_to_obj_state_fps_{}.json'.
                                                       format(int(self.fps)))
        self.time_to_obj_state_fps = load_json_dict(time_to_obj_state_fps_json_file)

        time_to_keypoint_fps_json_file = os.path.join(self.root_dir, 'annotations', 'time_to_keypoint_fps_{}.json'.
                                                      format(self.fps))
        self.time_to_keypoint_fps = load_json_dict(time_to_keypoint_fps_json_file)

        if bbox_gt:
            if train:
                time_to_bbox_file = os.path.join(self.root_dir, 'annotations', 'train_syn_time_to_bbox.json')
            else:
                time_to_bbox_file = os.path.join(self.root_dir, 'annotations', 'test_syn_time_to_bbox.json')
            self.time_to_bbox = load_json_dict(time_to_bbox_file)
        else:
            self.time_to_bbox = None
        self.bbox_gt = bbox_gt

        if train:
            cleaned_start_states_json_file = os.path.join(self.root_dir, 'annotations', 'train_cleaned_start_states.json')
        elif args.use_val:
            cleaned_start_states_json_file = os.path.join(self.root_dir, 'annotations', 'val_cleaned_start_states.json')
        else:
            cleaned_start_states_json_file = os.path.join(self.root_dir, 'annotations', 'test_cleaned_start_states.json')
        self.cleaned_start_states = load_json_dict(cleaned_start_states_json_file)['timestamps']

        self.object_paths = {obj: os.path.join(self.root_dir, 'objects_16k', obj, 'google_16k', 'textured.urdf') for obj
                             in self.object_list}

        self.fps = args.fps
        self.subsample_rate = args.subsample_rate
        self.render = args.render
        self.scale = scale
        if train:
            assert args.dropout_ratio > 0
            self.transform = transforms.Compose([
                transforms.Scale((self.scale, self.scale)),
                # transforms.RandomGrayscale(p=0.3),
                transforms.ColorJitter(hue=.05, saturation=.05, brightness=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Scale((self.scale, self.scale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.image_size = self.scale
        self.sequence_length = args.sequence_length
        self.use_syn = args.use_syn

        print('Get object sequences.')
        self.all_possible_data = self.get_possible_object_sequences()
        self.final_obj2image = calculate_data_stats(self.all_possible_data)
        self.all_possible_data += self.get_possible_reverse_sequences()

        if environment is None:
            print('initialize multiprocessing environments')
            environment = SubprocPhysicsEnv(args=args, object_paths=self.object_paths, context='spawn', nproc=11, nenvs=44)
            environment.reset()
            print('done initializing multiprocessing environments')
        args.instance_environment = environment
        self.environment = environment

    # Just a data augmentation
    def get_possible_reverse_sequences(self):
        possibilities = []
        for item in self.all_possible_data:
            sequence = item['sequence'].copy()
            sequence.reverse()
            possibilities.append({
                'obj_name': item['obj_name'],
                'clip_ind': item['clip_ind'],
                'sequence': sequence,
            })
        return possibilities

    def get_unique_clip_index(self, sequence, obj_name):
        this_object_map = self.time_to_clip_ind_image_adr[obj_name]
        clip_indices = [this_object_map[time]['clip_ind'] for time in sequence if time in this_object_map]
        if len(clip_indices) < len(sequence) or len(set(clip_indices)) != 1:
            return None
        else:
            return clip_indices[0]

    def get_possible_object_sequences(self):
        possibilities = []
        for obj in self.object_list:

            all_times_for_this_obj_sorted = [get_time_from_str(k) for k in self.time_to_obj_state_fps[obj].keys()]
            all_times_for_this_obj_sorted.sort()

            for ind in range(len(all_times_for_this_obj_sorted) - self.sequence_length * self.subsample_rate):

                sequence = [str(all_times_for_this_obj_sorted[i])
                            for i in range(ind, ind + self.sequence_length * self.subsample_rate, self.subsample_rate)]

                clip_index = self.get_unique_clip_index(sequence, obj)

                if clip_index is None:  # This means there are some objects that do not belong
                    continue

                if sequence[0] not in self.cleaned_start_states:
                    continue

                # maybe move it so somewhere else? This is not optimized
                contact_point = self.clip_to_contact_point[obj][str(clip_index)]['contact_point']
                contact_point = torch.Tensor(contact_point)
                needs_pruning = (contact_point > 5) + (contact_point < -5) + (contact_point != contact_point)
                any_unvalid = needs_pruning.sum(-1) > 0
                if torch.any(any_unvalid):
                    pdb.set_trace()
                possibilities.append({
                    'obj_name': obj,
                    'clip_ind': clip_index,
                    'sequence': sequence,
                })

        return possibilities

    def __len__(self):
        return len(self.all_possible_data)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def __getitem__(self, idx):
        data_point_dict = self.all_possible_data[idx]
        obj_name = data_point_dict['obj_name']
        clip_ind = data_point_dict['clip_ind']
        sequence = data_point_dict['sequence']
        time_to_keypoint_this_object = self.time_to_keypoint_fps[obj_name]
        contact_point = torch.Tensor(self.clip_to_contact_point[obj_name][str(clip_ind)]['contact_point'])

        contact_point = scale_position(contact_point, obj_name)

        all_rotations = []
        all_translations = []
        all_images = []
        all_syn_images = []
        all_keypoints = []
        all_image_paths = []
        all_bbox = []

        time_to_clip_ind_image_adr_this_object = self.time_to_clip_ind_image_adr[obj_name]
        time_to_obj_state_fps_this_object = self.time_to_obj_state_fps[obj_name]
        for time in sequence:
            image_path = time_to_clip_ind_image_adr_this_object[time]['image_adr'].replace('LMJTFY/', '')
            this_item_dict = time_to_obj_state_fps_this_object[time]
            position = scale_position(this_item_dict['position'], obj_name)
            rotation = this_item_dict['rotation']
            real_img_path = os.path.join(self.root_dir, image_path)
            all_image_paths.append(real_img_path)
            rgb = self.load_and_resize(real_img_path)
            if self.use_syn:
                prefix, fname = os.path.split(real_img_path)
                syn_path = os.path.join(prefix.replace('images', 'synthetic'), 'syn_' + fname)
                syn_rgb = self.load_and_resize(syn_path)
                all_syn_images.append(syn_rgb)
            if self.bbox_gt:
                one_box = self.time_to_bbox[obj_name][time]['bbox']
                all_bbox.append(one_box)
            else:
                all_bbox.append([])
            all_rotations.append(rotation)
            all_translations.append(position)
            all_images.append(rgb)

            keypoint_annot = {}
            if time in time_to_keypoint_this_object:
                keypoint_annot = time_to_keypoint_this_object[time]
            all_keypoints.append(process_projection(keypoint_annot))

        all_rotations = torch.Tensor(all_rotations).float()
        all_keypoints = torch.Tensor(all_keypoints).float()
        all_bbox = torch.Tensor(all_bbox).long()
        all_translations = torch.stack(all_translations, dim=0).float()
        all_images = torch.stack(all_images, dim=0)
        if self.use_syn:
            all_syn_images = torch.stack(all_syn_images, dim=0)
        input = {
            'rgb': all_images,
            'syn_rgb': all_syn_images,
            'image_paths': all_image_paths,
            'initial_position': all_translations[0],
            'initial_rotation': all_rotations[0],
            'initial_keypoint': all_keypoints[0],
            'object_name': obj_name,
            'contact_points': contact_point,
            'timestamps': sequence,
            'initial_bbox': all_bbox[0],
        }

        labels = {
            'keypoints': all_keypoints[1:],
            'position': all_translations[1:],
            'rotation': all_rotations[1:],
            'contact_points': contact_point,
            'bboxes': all_bbox[1:],
        }

        return input, labels
