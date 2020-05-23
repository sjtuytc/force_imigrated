import os
import random
from utils.data_io import save_into_json


class NSDatasetManager:
    def __init__(self, args):
        self.args = args
        self.all_data = []
        self.dataset_size, self.dataset_p, self.save_freq = args.dataset_size, args.dataset_p, args.save_freq
        self.data_counter = 0
        os.makedirs(self.dataset_p, exist_ok=True)
        random.seed(0)

    def insert_one_data(self, input_d, output_d, is_batch=True):
        if self.data_counter >= self.dataset_size:
            return
        else:
            self.data_counter += 1
        if is_batch:
            input_d = random.choice(input_d)
            output_d = random.choice(output_d)
        self.all_data.append({'input': input_d, 'output': output_d})
        if len(self.all_data) == self.save_freq or self.args.debug:
            print("Data num:", self.data_counter, 'save path:', self.dataset_p)
            batch_idx = int(self.data_counter / self.save_freq)
            save_into_json(save_obj=self.all_data, folder=self.dataset_p, file_name=batch_idx, verbose=False)
        if len(self.all_data) == self.save_freq:
            self.all_data = []
        return
