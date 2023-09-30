![Image text](https://github.com/zhengyanzhao1997/COOP/blob/master/imgs/headpic.png)

This repo is the official implementation for the ICCV2023 paper "COOP: Decoupling and Coupling of Whole-Body Grasping Pose Generation".

COOP is a novel framework to synthesize life-like whole-body poses that cover the widest range of human grasping capabilities.
![Image text](https://github.com/zhengyanzhao1997/COOP/blob/master/imgs/coop.jpg)

For more details please refer to the paper <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_COOP_Decoupling_and_Coupling_of_Whole-Body_Grasping_Pose_Generation_ICCV_2023_paper.html" target="_blank">COOP: Decoupling and Coupling of Whole-Body Grasping Pose Generation</a>.

## Installation

To install the dependencies please follow the next steps:

- Clone this repository: 
    ```Shell
    git clone https://github.com/zhengyanzhao1997/COOP
    cd COOP
    ```
- Install the dependencies by the following command:
    ```
    pip install -r requirements.txt
    ```

## Getting started

#### BNet and HNet
- we have released the checkpoints of BNet and HNet in the folders as below.
```bash
    COOP
    ├── src
    │   ├── models
    │       ├── Bnet
    │       └── Hnet
```

#### Download SMPLX and MANO models
- Download body models following the steps on the [SMPLX repo](https://github.com/vchoutas/smplx).
- Download body models following the steps on the [MANO repo](https://mano.is.tue.mpg.de/).

- Please put the SMPLX and MANO models in the folders as below.
```bash
    COOP
    ├── model
    │   ├── smplx
    │   └── mano
```

### Generate whole-body grasps for test split.

    cd src

Then

    python infer/infer.py --rh-work-dir ../model/Hnet/ --body-work-dir../model/Bnet/  --data-file ../data --save-file [your save path]
    
You can also specify the test object name, subject shape and the position of the test obejct:

    python ... --object apple --subject s1 --x 0.4 --z 0.4 --y 0.4

The visualize result (.html) and the generated mesh file will be saved at your save path.

### Training your own COOP.
coming soon

## Citation
```
@InProceedings{Zheng_2023_ICCV,
    author    = {Zheng, Yanzhao and Shi, Yunzhou and Cui, Yuhao and Zhao, Zhongzhou and Luo, Zhiling and Zhou, Wei},
    title     = {COOP: Decoupling and Coupling of Whole-Body Grasping Pose Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {2163-2173}
}
```