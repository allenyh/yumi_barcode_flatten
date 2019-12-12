#!/usr/bin/env bash
xhost +local:docker

# New docker user
DOCKER_USER="developer"

COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[0;33m'
COLOR_NC='\033[0m'

#
# Check the command 'nvidia-docker' existed or not
#
ret_code="$(command -v nvidia-docker)"
if [ -z "$ret_code" ]; then
    DOCKER_CMD="docker"
else
    DOCKER_CMD="nvidia-docker"
fi


#
# Specify cuda version
#
if [ $# -gt 0 ]; then
    if [[ "$1" == "cuda9" || "$1" == "cuda9.0" ]] ; then
        echo -e "RUN: \"${DOCKER_CMD}\""
        DOCKER_TAG="cuda9.0"
    elif [[ "$1" == "cuda10" || "$1" == "cuda10.0" ]] ; then
        echo -e "RUN: \"${DOCKER_CMD}\""
        DOCKER_TAG="cuda10.0"
    elif [ "$1" == "same" ] ; then
        echo -e "RUN: \"docker exec\""
    else
        echo -e "Please specify which cuda version your GPU support."
        echo -e "${COLOR_RED}Usage: source docker_run.sh [cuda9 | cuda10 | same]${COLOR_NC}"
    fi
else
    echo -e "${COLOR_RED}Usage: source docker_run.sh [cuda9 | cuda10| same]${COLOR_NC}"
fi


#
# Execute command
#
if [ $# -gt 0 ]; then
    if [ "$1" == "same" ]; then
        docker exec -it yumi-barcode-flatten bash
    else
        ${DOCKER_CMD} run --name yumi-barcode-flatten --rm -it --net=host --privileged -v /dev:/dev \
            --env="DISPLAY"="$DISPLAY" \
	    --env="QT_X11_NO_MITSHM=1" \
            -v /etc/localtime:/etc/localtime:ro -v /var/run/docker.sock:/var/run/docker.sock \
            -v /home/$USER/yumi_barcode_flatten:/home/${DOCKER_USER}/yumi_barcode_flatten \
            --volume="/tmp/.X11-unix/:/tmp/.X11-unix:rw" \
	    -env="XAUTHORITY=$XAUTH" \
	    --volume="$XAUTH:$XAUTH" \
	    --runtime=nvidia \
            -w /home/${DOCKER_USER}/yumi_barcode_flatten \
            --device=/dev/dri:/dev/dri \
            --device=/dev/nvhost-ctrl \
            --device=/dev/nvhost-ctrl-gpu \
            --device=/dev/nvhost-prof-gpu \
            --device=/dev/nvmap \
            --device=/dev/nvhost-gpu \
            --device=/dev/nvhost-as-gpu \
            --device=/dev/ttyUSB0 \
            -v /dev/bus/usb:/dev/bus/usb \
            argnctu/yumi-project:${DOCKER_TAG}
    fi
else
    echo "please provide docker tag name."
fi
