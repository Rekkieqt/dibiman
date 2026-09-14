/* yarp libraries */
#include "yarp/os/Bottle.h"
#include "yarp/os/BufferedPort.h"
#include <yarp/os/Network.h>
#include <yarp/os/Time.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/ITorqueControl.h>
#include <yarp/dev/IControlMode.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/IAxisInfo.h>

/* cpp libraries */
#include <vector>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <list>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <filesystem>

/* Pinocchio libraries */

#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"


/* Casadi and Eigen */
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

/* NMPC libraries */
#include "dibiman/nmpc/centralizedNMPC.hpp"

using namespace yarp::os;
using namespace yarp::dev;
using namespace pinocchio;

/* logging function */
void logData(std::fstream & logfile, Eigen::Ref<Eigen::VectorXd> tau, Eigen::Ref<Eigen::VectorXd> qread, int size) {
  /* log torques */
  for (int i=0; i < size; ++i) {
    logfile << qread[i] << ",";
  }
  /* log torques */
  for (int i=0; i < size; ++i) {
    logfile << tau[i] << ",";
  }
  logfile << std::endl;
}

void logHeader(std::fstream & logfile, int size) {
  /* log readings */
  for (int i=0; i < size; ++i) {
    logfile << "joint_meas" << i << ",";
  }
  /* log torques */
  for (int i=0; i < size; ++i) {
    logfile << "torque " << i << ",";
  }
  logfile << std::endl;
}

void logRef(std::fstream & reffile, Eigen::VectorXd& qref, int size) {
  /* log refs */
  for (int i=0; i < size; ++i) {
    reffile << "joint_ref" << i << ",";
  }
  reffile << std::endl;
  /* log torques */
  for (int i=0; i < size; ++i) {
    reffile << qref[i] << ",";
  }
  reffile << std::endl;
}


int main(int argc, char **argv)
{
    /* dynamic parameter loading with rf*/
    ResourceFinder rf;
    rf.configure(argc, argv);

    /* findGroup -> find a list and save it as a Bottle, useful for later */
    std::string robotName = rf.find("robot").asString();
    std::string partName = rf.find("part").asString();
    std::string local = rf.find("local").asString();
    std::string remote = rf.find("remote").asString();
    std::string comm = rf.find("comm").asString();
    std::string arm_prefix = rf.find("prfx").asString();
    std::string urdf_filename = rf.find("model").asString(); 

    int joints = rf.find("joints").asInt32();
    bool single_mode = rf.find("single").asBool();
    
    if (robotName=="")
    {
        std::cout << "Failed loading config file!" << std::endl;
        return -1;
    }

    /* ++++++ YARP NETWORKING ++++++ */

    /* configuring in and out ports */
    /* where to connect */
    std::string remotePorts="/";
    remotePorts+=robotName;
    remotePorts+="/";
    remotePorts+=partName;

    /* name of local port */
    std::string localPorts= rf.find("local").asString();

    /* connect to simulated control board*/
    Property options;
    options.put("device", "remote_controlboard");
    options.put("local", localPorts.c_str());   
    options.put("remote", remotePorts.c_str()); 

    /* create a device */
    PolyDriver robotDevice(options);
    if (!robotDevice.isValid()) {
        yError("Device not available.\n");
        return 0;
    }

    /* create interfaces */
    IControlMode *controlMode;
    ITorqueControl *torqueControl;
    IPositionControl *positionControl;
    IEncoders *armSensors;
    IAxisInfo *axInfo;

    /* check if interfaces are available */
    bool ok;
    ok = robotDevice.view(controlMode);
    ok = ok && robotDevice.view(torqueControl);
    ok = ok && robotDevice.view(positionControl);
    ok = ok && robotDevice.view(armSensors);
    ok = ok && robotDevice.view(axInfo);
    
    if (!ok) {
        yError("Problems acquiring interfaces\n");
        return 0;
    }

    /* establish comms between right and left arm */
    /* from where to write */
    std::string inPort = rf.find("inport").asString();
    /* where to receive */
    std::string outPort = rf.find("outport").asString();
    /* where to write */
    std::string writePort = rf.find("writeport").asString();

    BufferedPort<Bottle> recvPort;
    BufferedPort<Bottle> sendPort;

    recvPort.open(inPort);
    sendPort.open(outPort);

    /* yarp networking */
    Network yarp;

    /* wait for the other node to come online */
    while (!(yarp.connect(outPort, writePort) || single_mode)) {
        std::cout << "Waiting for the other node to connect ..." << std::endl;
        yarp::os::Time::delay(3);
    }

    /* +++++++++++ ARM CONTROL START ++++++++++++++*/
    /* arm joints */
    const std::vector<std::string> joint_list = {"shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_prosup", "wrist_pitch", "wrist_yaw"};
    const std::vector<std::string> ids = {"right_icub_arm", "left_icub_arm"};
    const std::vector<std::string> end_effector_frame_names = {"r_hand", "l_hand"};
    const std::string urdf_path = static_cast<std::string>(std::filesystem::current_path()) + "/../conf/model.urdf";

    std::vector<std::string> right_arm_joint_list;
    std::vector<std::string> left_arm_joint_list;

    for(auto it = joint_list.begin();
            it != joint_list.end();
            ++it)
    {
        right_arm_joint_list.push_back("r_" + *it);
        left_arm_joint_list.push_back("l_" + *it);
    }

    std::vector<std::vector<std::string>> list_of_joints;
    list_of_joints.push_back(right_arm_joint_list);
    list_of_joints.push_back(left_arm_joint_list);

    /* nmpc object init */
    dibiman::centralizedNMPC manip_controller(
        urdf_path,
        list_of_joints,
        end_effector_frame_names,
        ids
        );

    /* remote controller */
    int idx_joints[] = {0, 1, 2, 3, 4, 5, 6};
    int all_arm_joints = 0;
    armSensors->getAxes(&all_arm_joints);

    /* Controlled joints */
    double* q_sens = new double[joints]; /* joint positions */
    // double* v_sens = new double[joints]; /* joint velocities */
    // double* a_sens = new double[joints]; /* joint accelerations */
    // double* u_sens = new double[joints]; /* tau -> torque at the joints */ 

    /* dynamic link between double arrays and eigen arrays */
    Eigen::Map<Eigen::VectorXd> q_sens_Vec(q_sens, joints);
    // Eigen::Map<Eigen::VectorXd> qd_meas_v(qd_all, joints);
    // Eigen::Map<Eigen::VectorXd> qdd_meas_v(qdd_meas, joints);
    // Eigen::Map<Eigen::VectorXd> tau_v(tau, joints);
    // Eigen::Map<Eigen::VectorXd> tau_meas_v(tau_meas, joints);

    std::string logfile = "../../logs/" + partName + ".csv";
    std::fstream file(logfile, std::ios::out | std::ios::trunc);
    logHeader(file, joints);

    /* set control mode for arm */
    // int modes[] = {VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE};
    // controlMode->setControlModes(joints, idx_joints, modes);

    /* log the ref angles */
    // std::string reffile = "../../logs/" + partName + "_ref.csv";
    // fstream ref(reffile, ios::out | ios::trunc);
    /* logRef(ref, q_ref_v, joints); */
    // ref.close();

    for (int i = 0; i < 2; i++) {

        for (int j = 0; j < static_cast<int>(std::size(idx_joints)) ; j++) {
          armSensors->getEncoder(idx_joints[j], &q_sens[j]);
          q_sens[j] = (M_PI/180) * q_sens[j];
          std::cout << q_sens[j] << std::endl;
        }
        // pinocchio::forwardKinematics(reduced_model, data, q_sens_Vec);  
        // pinocchio::updateFramePlacements(reduced_model, data);  
        // std::cout << data.oMf[handID] << std::endl;

        /* send torque commands */
        // ok = torqueControl->setRefTorques(joints, idx_joints, tau);

        /* write to file */
        // logData(file, tau_v, q_meas_v, joints);

        /* send info to other node(arm) */
        Bottle *b = recvPort.read(false);
        if (b!=NULL) {
            /* receive communication works */
            // cout << "Data received!\n";
            // cout << "got " << b->toString().c_str();
        }
        Bottle& s = sendPort.prepare();
        s.clear();
        s.addString(inPort);
        sendPort.write();
        
        yarp::os::Time::delay(0.05);
    }
    /* cleanup, effectively useless because at the moment I ctrl+c from while */
    /* later can put this into 'graceful' exit with interrupt ... */

    robotDevice.close();
    file.close();
    
    return 0;
}
