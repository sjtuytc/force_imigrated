## Sec1: Installation

1. create an environment

```shell
conda create -n force python=3.7
```

2. download datasets

```shell
git clone https://github.com/chentinghao/download_google_drive.git

# download dataset
python download_google_drive/download_gdrive.py 11HxjPTHqjLncMSxOg7ERZWRzPNKst4Su ./force.tar.gz

# download pretrained weight
python download_google_drive/download_gdrive.py 149s4eUJn9owuXSm0Y5_5ffX5bGOCXTeh /dev/shm/dataset/force_cvpr2020_trained_weights.tar.gz

python download_google_drive/download_gdrive.py 1dRdC7sTz7FAY_iiori0UGZ1gDCdE9nEz ./force_code.tar.gz

# unzip downloaded dataset
tar xvzf force.tar.gz
unzip images_zip.zip
tar xvzf force_cvpr2020_trained_weights.tar.gz
```

3. install libraries

```shell
pip install -r requirements.txt
```

## Sec2: training/testing

1. Training command:

```shell
# train the full model
python main.py --title joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train

# with batch processing
python main.py --title joint_training --sequence_length 10 --gpu-ids -1 --number_of_cp 5 --model BatchSeparateTowerModel --dataset BatchDatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train

python debug/test_physics_env.py --title joint_training --sequence_length 10 --gpu-ids -1 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train

python test_physics_env.py --title joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train
# given gt cp predict force
python3 main.py --title train_all --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list ALL --data DatasetForce
```

```shell
python main.py --title train_cp_prediction --batch-size 64 --workers 10 --gpu-ids 0 --number_of_cp 5 --model NoForceOnlyCPModel --dataset DatasetWAugmentation --loss CPPredictionLoss --object_list ALL --break-batch 1 --data DatasetForce
```

2. Testing command:

```shell
# test all based on predicted
python main.py --title test_all_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_end2end.pytar test
```

```shell
# test joint training.
python3 main.py --title test_all_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_end2end.pytar test
```

3. debugging command:

```shell
python debug/test_subproc_physics_env.py --title joint_training --sequence_length 10 --gpu-ids -1 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train
```

```shell
python debug/test_physics_env.py --title joint_training --sequence_length 10 --gpu-ids -1 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train
```



