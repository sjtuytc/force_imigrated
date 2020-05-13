import torch
from utils.data_loading_utils import load_json_dict
from .keypoint_and_trajectory_dataset import DatasetWAugmentation


class BaselineForceDatasetWAugmentation(DatasetWAugmentation):
    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):
        super(BaselineForceDatasetWAugmentation, self).__init__(args, train)
        predicted_force_dict_adr = args.predicted_cp_adr
        self.predicted_force_dict = load_json_dict(predicted_force_dict_adr)
        self.train = train

    def get_possible_reverse_sequences(self):
        return []

    def __getitem__(self, idx):
        input, labels = super(BaselineForceDatasetWAugmentation, self).__getitem__(idx)

        if self.train:
            item = self.all_possible_data[idx]
            sequence = item['sequence']
            sequence_str = '__'.join(sequence)
            if sequence_str not in self.predicted_force_dict.keys():
                print(sequence_str, "not found!")
                raise RuntimeError
                labels['forces'] = torch.zeros(self.sequence_length - 1, self.number_of_cp, 3)
            else:
                forces = torch.Tensor(self.predicted_force_dict[sequence_str])
                labels['forces'] = forces
        else:
            labels['forces'] = torch.zeros(self.sequence_length - 1, self.number_of_cp, 3)

        return input, labels
