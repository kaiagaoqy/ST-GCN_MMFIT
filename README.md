# STGCN-SWMV

This is the code, data repository and pre-trained models for our work :

## Introduction

We propose to tackle the online action recognition problem with a sliding window and majority voting approach using Spatial Temporal Graph Convolutional Neural Networks.
Using only skeletal data, we first consider the 3D coordinates of each joint of the human skeleton as a chain of motion and represent them as a graph of nodes. The spatial and temporal evolution of this characteristic vector is then represented by a trajectory in the space of actions allowing us to simultaneously capture both geometrical appearances of the human body and its dynamics over time. The online action recognition problem is then formulated as a problem of finding similarities between the shapes of trajectories over time.

## Pre-requisites & Installation

Install all required libraries by running :

``` shell
pip install -r requirements.txt
cd torchlight & python setup.py install & cd ..
```
## Demo
We provided demo file for trainning and testing our model
```
jupyter Test.IPYNB
```

## Dataset 
- (MMFit)[https://mmfit.github.io/]

  
## Download data and pre-trained models
We provided the data and the pre-trained models of our **STGCN-SWMV** method for the OAD and UOW datasets. To download them, please run these scripts :
```
bash tools/rsc/get_data.sh
bash tools/rsc/get_weights.sh
```

**For Windows users :**

First, download WGet.exe from this link : [WGet](https://eternallybored.org/misc/wget/1.20.3/64/wget.exe) and copy it to the Windows/System32 directory.
Then open bash files with [GIT](https://git-scm.com/download/win).

## Work in progress
- [x] Action detection based on sliding windows and 3D skeleton dataset
- [x] Process real-time video and output result
- [ ] Add Rep-Recognizatoin module (Detect completeness)
- [x] Data-Proprocess: transforms input data into the relative coordinate system with the center-of-the-spine joint as the origin.
- [ ] Add completeness and angle infomation into GCN


## Results

Here are our results using the STGCN-SWMV method on the MMFIT online skeleton-based datasets.

**MMFIT:**
<p align="center">
	<img src="rsc/MMFIT Confusion Matrix.png" alt="MMFIT Confusion Matrix">
</p>

| Actions | Results | 
|:-------:|:-------:|
| Squat | 1.000|
| Lunge | 0.998|
| Bicep Curl | 0.999 |
| Sit up| 1.000|
| Pushup| 1.000|
| Tricep Extension| 1.000|
|dumbbell_row| 1.000|
|jumping jack|1.000|
|dumbbell shoulder press|1.000|
|lateral shoulder raise|1.000|
| **Overall** | **0.9996** |


## Test models

To test the **STGCN-SWMV** method and replicate our results, please run :

**For the MMFIT dataset :**

```python main.py stgcn_swmv --dataset=MMFIT --use_gpu=True -c config/stgcn_swmv/MMFIT/test.yaml```


## Training from scratch

To train the **STGCN-SWMV** method from the scratch, please run :

**For the MMFIT dataset :**

```python main.py stgcn_swmv --dataset=MMFIT --use_gpu=True -c config/stgcn_swmv/MMFIT/train.yaml```


**NOTE** : If --use_gpu is set to true make sure you have installed [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) for GPU acceleration.
Set --use_gpu to False if you want to use CPU instead.
Make sure also to set the ```device``` parameter in the .yaml config files situated in config/stgn_swmv/Dataset_name/train.yaml to the number of GPUs on your computer.
If you encountred memory errors try to reduce the ```batch_size```.

If you any questions or problems regarding the code, please contact us at : <qgao14@jhu.edu>.

## Citation
To cite this work, please use:
``` 
Citation will be updated once our paper get accepted.
```
