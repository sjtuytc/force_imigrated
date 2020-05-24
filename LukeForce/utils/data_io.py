import os
import pickle
import json


def save_into_pkl(save_obj, full_path=None, name="test",
                  folder="./debug", verbose=False):
    if full_path is None:
        full_path = os.path.join(folder, str(name) + '.pkl')
    output = open(full_path, 'wb')
    pickle.dump(save_obj, output)
    output.close()
    if verbose:
        print("Current obj saved at", os.path.join(full_path))
    return full_path


def read_from_pkl(name="test", folder="./debug", full_path=None):
    if full_path is None:
        full_path = os.path.join(folder, str(name) + ".pkl")
    pkl_file = open(full_path, 'rb')
    return_obj = pickle.load(pkl_file)
    pkl_file.close()
    return return_obj


def save_into_json(save_obj, folder=None, file_name="test", full_path=None, verbose=True):
    if full_path is None:
        full_path = os.path.join(folder, str(file_name) + ".json")
    gt_file = open(full_path, 'w', encoding='utf-8')
    json.dump(save_obj, gt_file)
    if verbose:
        print("Current obj saved at", full_path)
    gt_file.close()
    return full_path


def read_from_json(folder=None, file_name="test", full_path=None, verbose=False):
    if full_path is None:
        full_path = os.path.join(folder, str(file_name) + ".json")
    file_obj = open(full_path)
    data_obj = json.load(file_obj)
    file_obj.close()
    if verbose:
        print("Read obj from", full_path)
    return data_obj
