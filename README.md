# yumi_barcode_flatten
The repo is prepared for YUMI yumi_barcode_flatten project.

## Prerequisite
 1. Ubuntu 16.04/18.04 with Nvidia driver >= 384.xx (supprt cuda9.0 or newer)
 2. [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) & [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker#upgrading-with-nvidia-docker2-deprecated)

## Setup 
a. Clone repo in $HOME directory
```bash
    $ cd ~/ && git clone https://github.com/allenyh/yumi_barcode_flatten.git
    $ git submodule init && git submodule update
```

b. Docker run 
```bash
    # It still can work for ROS stuff if your laptop doesn't support cuda10.
    $ docker pull argnctu/yumi-project:cuda10.0    # To check your docker image is latest version
    $ cd yumi_barcode_flatten
    $ source docker_run.sh [cuda10 | same]    # tag"same" is for "docker exec" command
```

c. Compile the code **in docker container**
```bash
    $ cd ~/yumi_barcode_flatten/catkin_ws
    $ catkin_make
```

___

## How to run
### TODO Write






