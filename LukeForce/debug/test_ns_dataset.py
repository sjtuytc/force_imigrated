from datasets.ns_dataset import NSDataset


def test_dataset():
    root_dir = "NSDataset_v1/"
    train_d = NSDataset(root_dir=root_dir, train_val_rate=0.9, train=True)
    asd = train_d[0]
    for k in enumerate(train_d):
        print(len(k))
        i=1
    more_d = train_d[125]
    val_d = NSDataset(root_dir=root_dir, train_val_rate=0.9, train=False)
    print(len(val_d))


if __name__ == '__main__':
    test_dataset()
