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

## Sec3: training/testing

1. Training command of LukeForce:

```shell
# train the seperate NS/FP model.
python main.py --title v7_first_feature_lstm --gpu-ids 3 --environment PhysicsEnv --model SeperateFPAndNS --dataset DatasetWAugmentation --loss SeperateFPNSLoss --object_list ALL --data DatasetForce --batch-size 1 --loss1_w 1.0 --loss2_w 1.0 --set-gpu-ids-in-env --break-batch 32 --use_gt_cp train

# train the joint ns model.
srun --gres=gpu:2 -w SH-IDC1-10-198-6-147 python main.py --title v5_loss1_only_no_scale --gpu-ids 0 --environment PhysicsEnv --model JointNS --dataset DatasetWAugmentation --loss JointNSProjectionLoss --object_list ALL --data DatasetForce --batch-size 1 --loss1_w 1.0 --loss2_w 0.0 train

# update two losses at the same time (use vis_grad to visualize gradients)
srun --gres=gpu:2 -w SH-IDC1-10-198-6-147 python main.py --title v5_use_gt_cp_1_plus_1 --gpu-ids 0 --environment PhysicsEnv --model JointNS --dataset DatasetWAugmentation --loss JointNSProjectionLoss --object_list ALL --data DatasetForce --batch-size 1 --loss1_w 1.0 --loss2_w 1.0 --joint_two_losses --use_gt_cp --vis_grad train

# train the full model (original, no batch processing)
python main.py --title original_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train

# train the full model with batch processing
python main.py --title batch_joint_training_v3 --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --environment NpPhysicsEnv --model BatchSeparateTowerModel --dataset BatchDatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train

# only predict contact point.
python main.py --title train_cp_prediction --batch-size 1 --workers 10 --gpu-ids 0 --number_of_cp 5 --model NoForceOnlyCPModel --dataset DatasetWAugmentation --loss CPPredictionLoss --object_list ALL --break-batch 1 --data DatasetForce train

# Calculate and save gt force.
python main.py --title save_gt_force --batch-size 1 --workers 10 --gpu-ids 1 --number_of_cp 5 --model NoModelGTForceBaseline --dataset BaselineForceDatasetWAugmentation --loss KeypointProjectionLoss --object_list ALL --break-batch 1 --data DatasetForce --predicted_cp_adr DatasetForce/gtforce_train.json savegtforce
```

2. Testing command:

```shell
# test joint ns model.
python main.py --title loss1onlycheck --reload loss1onlycheck.pytar --gpu-ids 0 --environment PhysicsEnv --model JointNS --dataset DatasetWAugmentation --loss JointNSProjectionLoss --object_list ALL --data DatasetForce --batch-size 1 --loss1_w 1.0 --loss2_w 1.0 --vis test

# test the base model of NS.
python main.py --title train_ns_v3 --reload model_state.pytar --sequence_length 10 --ns --gpu-ids 0 --number_of_cp 5 --model NSBaseModel --dataset NSDataset --loss StateEstimationLoss --object_list ALL --data DatasetForce --batch-size 1 --break-batch 1 --epochs 1000  --save_frequency 30  --ns_dataset_p NSDatasetV5 --vis test

# test all based on predicted
python main.py --title test_all_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_end2end.pytar test
```

```shell
# test joint training.
python3 main.py --title test_all_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_end2end.pytar test
```

3. debugging command:

```shell
# test model
python debug/test_model.py --title debug_model  --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model BatchCPHeatmapModel --dataset BatchDatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train

# test trained model
python debug/test_model.py --title test_all_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_end2end.pytar test
# test physics env (no multiprocessing)
python debug/test_physics_env.py --title joint_training --sequence_length 10 --gpu-ids -1 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train
# test multiprocessing physics env
python debug/test_subproc_physics_env.py --title joint_training --sequence_length 10 --gpu-ids -1 --environment NpPhysicsEnv --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train
# test multiprocessing physics env + collect dataset.
python debug/test_subproc_physics_env.py --title joint_training --sequence_length 10 --gpu-ids -1 --environment NpPhysicsEnv --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 --save_dataset train
# test physical layer.
python debug/test_physics_layer.py --title joint_training --sequence_length 10 --gpu-ids -1 --environment NpPhysicsEnv --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --batch-size 1 train
```

## Supple Sec

```shell
git fetch --all
git reset --hard origin/master
git pull
```

