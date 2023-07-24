
This repo is the official implementation for the ICCV2023 paper "COOP: Decoupling and Coupling of Whole-Body Grasping Pose Generation".

COOP is a novel framework to synthesize life-like whole-body poses that cover the widest range of human grasping capabilities.

For more details please refer to the paper.

## Installation

To install the dependencies please follow the next steps:

- Clone this repository: 
    ```Shell
    git clone https://github.com/zhengyanzhao1997/COOP
    cd GOAL
    ```
- Install the dependencies by the following command:
    ```
    pip install -r requirements.txt
    ```

## Getting started

#### Download SMPLX and MANO models
- Download body models following the steps on the [SMPLX repo](https://github.com/vchoutas/smplx).
- Download body models following the steps on the [MANO repo](https://mano.is.tue.mpg.de/).

- Please put the SMPLX and MANO models in the folders as below.
```bash
    COOP
    ├── src
    │   ├── models
    │       ├── smplx
    │       └── mano
```

#### Generate whole-body grasps for test split.

    cd src

Then

    python infer/infer.py --rh-work-dir ../model/Hnet/ --body-work-dir../model/Bnet/  --data-file ../data --save-file [your save path]
    
You can also specify the test object name, subject shape and the position of the test obejct:

    python ... --object apple --subject s1 --x 0.4 --z 0.4 --y 0.4

The visualize result (.html) and the generated mesh file will be saved at your save path.

## Citation
To be completed