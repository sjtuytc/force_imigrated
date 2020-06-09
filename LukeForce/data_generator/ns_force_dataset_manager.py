import os
import random
from utils.data_io import save_into_json


class NSDatasetManager:
    def __init__(self, args):
        self.args = args
        self.all_data = {}
        self.dataset_size, self.ns_dataset_p, self.save_freq = args.dataset_size, args.ns_dataset_p, args.save_freq
        self.data_counter = 0
        os.makedirs(self.ns_dataset_p, exist_ok=True)
        random.seed(0)

    def insert_one_data(self, input_d, output_d, is_batch=True, check_succ=True):
        if self.data_counter >= self.dataset_size:
            return
        if is_batch:
            input_d = random.choice(input_d)
            output_d = random.choice(output_d)

        if check_succ:
            all_succ = output_d['succ']
            succ_indicator = min(all_succ)
            if succ_indicator < 0.5:
                return
        obj_name = input_d['initial_state']['object_name']

        if obj_name in self.all_data.keys():
            self.all_data[obj_name].append({'input': input_d, 'output': output_d})
        else:
            self.all_data[obj_name] = []
        self.data_counter += 1
        for obj_name in self.all_data.keys():
            if self.data_counter % self.save_freq == 0 or self.args.debug:
                full_path = save_into_json(save_obj=self.all_data[obj_name], folder=self.ns_dataset_p, file_name=obj_name,
                                           verbose=False)
                print("Data num:", len(self.all_data[obj_name]), 'obj name:', obj_name)
        return
