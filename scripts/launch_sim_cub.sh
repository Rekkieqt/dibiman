#!/bin/bash

# Cleanup function
cleanup() {
    yarp clean
    kill $YARPSERVER_PID $GZSERVER_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Export variables
export ICUB_MODEL_PATH="/home/ubuntu/dibiman/conf/model.urdf"
export YARP_QUIET=1

# Script variables
GAZEBO_WORLD_PATH="/home/ubuntu/gazebo_models/worlds/icub-gazebo-world.sdf"
LEFT_ARM_DOT_INI_PATH="/home/ubuntu/dibiman/conf/left_arm.ini"
RIGHT_ARM_DOT_INI_PATH="/home/ubuntu/dibiman/conf/right_arm.ini"
BINARY_PATH="/home/ubuntu/dibiman/build/biman"

# Start servers
yarpserver --write --silent &
YARPSERVER_PID=$!

if [[ $1 ]]; then
  gazebo --minimal_comms $GAZEBO_WORLD_PATH &
  GZSERVER_PID=$!
else 
  gzserver --minimal_comms $GAZEBO_WORLD_PATH &
  GZSERVER_PID=$!
fi

# Wait for initialization
sleep 3

# Run binary
${BINARY_PATH} "--from" $LEFT_ARM_DOT_INI_PATH

# Cleanup is automatic via trap

