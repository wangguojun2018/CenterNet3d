## CenterNet3D: An Anchor free Object Detector for Autonomous Driving (Arxiv 2020) [\[paper\]](https://arxiv.org/abs/2007.07214)
Based on the center point, we propose an anchor-free CenterNet3D Network that performs 3D object detection without anchors. 
Our CenterNet3D uses keypoint estimation to find center points and directly regresses 3D bounding boxes. 
Besides, our CenterNet3D is Non-Maximum Suppression free which makes it more efficient and simpler. On the KITTI benchmark, 
our proposed CenterNet3D achieves competitive performance with other one stage anchor-based methods.

## Updates
2020-08-26: CenterNet3D V1.1 is released!

We develop an efficient keypoint-sensitive warping operation to align the confidences to the predicted bounding boxes

## Performance in KITTI validation set (50/50 split)
```centernet3d.py```(epochs 25,batch size 2):

```
Car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:90.65, 89.55, 88.85
bev  AP:89.98, 87.99, 86.98
3d   AP:89.02, 79.11, 77.76
aos  AP:90.63, 89.39, 88.62
Car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:90.65, 89.55, 88.85
bev  AP:90.66, 89.76, 89.28
3d   AP:90.66, 89.72, 89.20
aos  AP:90.63, 89.39, 88.62
```

## Demo
[![Demo](https://github.com/wangguojun2018/CenterNet3d/blob/master/demo/example1.png)](https://www.bilibili.com/video/BV1W541147gH/)

# Introduction
![model](https://github.com/wangguojun2018/CenterNet3d/blob/master/demo/Outline_of_CenterNet3D.png)  
Accurate and fast 3D object detection from point clouds is a key task in autonomous driving. Existing one-stage 3D object detection methods can achieve real-time performance, however, they are dominated by anchor-based detectors which are inefficient and require additional post-processing. In this paper, we eliminate anchors and model an object as a single point the center point of its bounding box. Based on the center point, we propose an anchor-free CenterNet3D Network that performs 3D object detection without anchors. Our CenterNet3D uses keypoint estimation to find center points and directly regresses 3D bounding boxes. However, because inherent sparsity of point clouds, 3D object center points are likely to be in empty space which makes it difficult to estimate accurate boundary. To solve this issue, we propose an auxiliary corner attention module to enforce the CNN backbone to pay more attention to object boundaries which is effective to obtain more accurate bounding boxes. Besides, our CenterNet3D is Non-Maximum Suppression free which makes it more efficient and simpler. On the KITTI benchmark, our proposed CenterNet3D achieves competitive performance with other one stage anchor-based methods which show the efficacy of our proposed center point representation.  

# Installation
1. Clone this repository.
2. Our CenterNet3D is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), Please check [INSTALL.md](https://github.com/wangguojun2018/CenterNet3d/blob/master/docs/install.md) for installation instructions.

# Train
To train the CenterNet3D, run the following command:
```
cd CenterNet3d
python tools/train.py ./configs/centernet3d.py
```

# Eval
To evaluate the model, run the following command:
```
cd CenterNet3d
python tools/test.py ./configs/centernet3d.py ./work_dirs/centernet3d/epoch_25.pth
```
## Citation
If you find this work useful in your research, please consider cite:
```
@misc{wang2020centernet3dan,
    title={CenterNet3D:An Anchor free Object Detector for Autonomous Driving},
    author={Guojun Wang and Bin Tian and Yunfeng Ai and Tong Xu and Long Chen and Dongpu Cao},
    year={2020},
    eprint={2007.07214},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgement
The code is devloped based on mmdetection3d and mmdetecton, some part of codes are borrowed from SECOND and PointRCNN.  
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 
* [mmdetection](https://github.com/open-mmlab/mmdetection) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN)
