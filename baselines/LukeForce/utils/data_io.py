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
