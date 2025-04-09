# SnapPose

SnapPose is a diffusion-based model 2D-3D pose estimation lifting. This README provides instructions for setup, training, testing, and inference.

## 1. Setup

### Environment Setup 

You can find all the packages and dependencies in the environment.yml file. 
If you have conda, you can simply run

```
conda env create -f environment.yml

```

### Download the Webdatasets

For simplicity you can put the websdatasets in the webdataset folder. You can store the webdatasets whethere you like, but ensure to specify the right path when launching the scripts.

### Download Pretrained Weights
Before starting training or testing, you need to download the pretrained weights of PoseHRNet and place them in the following folder:

```
checkpoint/posehrnet
```
Make sure to also download and place the ContextPoseFormer weights (human36m_nips_retrain.bin and mpi-inf_nips_retrain.bin) in the checkpoint folder.


## 2. Training the Model

To start training, run the following commands

for Human36m:

```
python main_diffusers_h36m.py --config config/human36m_train.yaml -w WDS_PATH -e 50 -c checkpoint/human36m 
```

for mpi-inf-3dhpe:

```
python main_diffusers_mpi.py --config config/human36m_train.yaml -w WDS_PATH -e 50 -c checkpoint/mpi-inf-3dhpe 
```


### Explanation of Parameters:
- `--config config/human36m_train.yaml` : Specifies the configuration file. It is the same for both datasets
- `-e 50` : Number of training epochs.
- `-c checkpoint` : Directory to save model weights.
- `-w WDS_PATH` : Path to the webdataset. For example /home/webdataset/Human36m

## 3. Testing the Model

To test the model, download the corresponding pretrained weights and place them in the folder:

```
checkpoint/human36m or checkpoint/mpi-inf-3dhpe
```
For Human36m:

```
python main_diffusers_h36m.py --config config/human36m_train.yaml -c checkpoint/human36m -w WDS_PATH \
                                                        --evaluate best_h36m.bin -num_proposals 20 -timesteps 20 
```

For mpi-inf-3dhpe:

```
python main_diffusers_mpi.py --config config/human36m_train.yaml -c checkpoint/mpi-inf-3dhpe -w WDS_PATH \
                                                       --evaluate best_mpi.bin -num_proposals 20 -timesteps 20\
```

### Explanation of Parameters:
- `--config config/human36m_train.yaml` : Specifies the configuration file. It is the same for both datasets
- `-c checkpoint` : Directory where model weights are stored.
- `-timesteps 20` : Number of diffusion steps.
- `-num_proposals 20` : Number of hypotheses generated for the image.
- `--evaluate best_{DATASET}.bin` : Loads the best model weights for the specified dataset.
- `-w WDS_PATH` : Path to the webdataset. For example /home/webdataset/Human36m


---


