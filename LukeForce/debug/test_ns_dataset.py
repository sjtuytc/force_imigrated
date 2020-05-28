from datasets.ns_dataset import NSDataset


def test_dataset():
    root_dir = "NSDataset_v2/"
    train_d = NSDataset(obj_name='019_pitcher_base', root_dir=root_dir, train_val_rate=0.9, train=True, data_statistics=None)
    asd = train_d[0]
    for k in enumerate(train_d):
        print(len(k))
        i=1
    more_d = train_d[125]
    # val_d = NSDataset(root_dir=root_dir, train_val_rate=0.9, train=False)
    # print(len(val_d))


if __name__ == '__main__':
    test_dataset()
