#!/bin/bash

# Cleanup function
cleanup() {
    yarp clean
    kill $YARPSERVER_PID $GZSERVER_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Export variables
export DIBIMAN_ROOT="/home/ubuntu/dibiman"
export ICUB_MODEL_PATH="$DIBIMAN_ROOT/conf/model_FT.urdf"
export YARP_QUIET=1

# Script variables
GAZEBO_WORLD_PATH="$DIBIMAN_ROOT/gazebo/grasp-world/grasp-world.sdf"
LEFT_ARM_DOT_INI_PATH="$DIBIMAN_ROOT/conf/left_arm.ini"
RIGHT_ARM_DOT_INI_PATH="$DIBIMAN_ROOT/conf/right_arm.ini"
BINARY_PATH="$DIBIMAN_ROOT/build/gazebo_simulations/gz-cub-single"

# Start servers
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$DIBIMAN_ROOT/gazebo
yarpserver --write --silent &
YARPSERVER_PID=$!

if [[ $1 ]]; then
  gz sim $GAZEBO_WORLD_PATH &
  GZSERVER_PID=$!
else 
  gz sim -s $GAZEBO_WORLD_PATH &
  GZSERVER_PID=$!
fi

# Wait for initialization
sleep 5

# Run binary
${BINARY_PATH} # "--from" $LEFT_ARM_DOT_INI_PATH

# Cleanup is automatic via trap
