# DROID-SLAM
Before : This README.md file refer to this [github](https://github.com/princeton-vl/DROID-SLAM) and change to suit my taste

## Requirements
To run the code you will need
* **Inference:** Running the demos will require a GPU with at least 11G of memory. 
* **Training:** Training requires a GPU with at least 24G of memory. We train on 4 x RTX-3090 GPUs.

## Getting Started
1. Clone the repo using the `--recursive` flag
    ```bash
    cd && git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
    ```

2. Creating a new anaconda environment
    ```bash
    conda create -n droid python=3.10
    echo 'alias cad="conda activate droid"' >> ~/.bashrc
    sb && cad
    pip install --upgrade pip setuptools
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    pip install evo --upgrade --no-binary evo
    pip install gdown
    pip install opencv-python torch-scatter open3d tqdm scipy tensorboard matplotlib pyyaml
    ```

3. Compile the extensions (takes about 10 minutes)
    ```bash
    cd DROID-SLAM
    wget https://raw.githubusercontent.com/j-wye/droid_slam/main/monocular.py
    wget https://raw.githubusercontent.com/j-wye/droid_slam/main/serial.py
    python setup.py install
    ```

## For My Project
- If you want to change rosbag file to png : check this python file [🔗](./rosbag_to_png.py)
1. Turtlebot4 DROID-SLAM with fastrtps
    ```bash
    cad && cd ~/DROID-SLAM
    python serial.py
    ```
    if you have errors about anaconda3 during ros2 communication, check [here](https://github.com/j-wye/j-wye.github.io/blob/main/issue/READEME.md)
    
    *<span style="color:Red">From here, progress things are going to write</spawn>*

2. Using Monocular Camera
    ```bash
    cad && cd ~/DROID-SLAM
    python monocular.py
    ```
    Test is not finished when using Depth camera or RGBD

## [Demos Original](./README_original.md)