/* yarp libraries */
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Network.h>
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

int main(int argc, char **argv)
{
    /* findGroup -> find a list and save it as a Bottle, useful for later */
    std::string right_robotName = "icubSim";/* icubSim */
    std::string right_partName = "right_arm" /* {ARM}_arm */
    std::string right_local = "/right_arm_controller"; /* /{ARM}_arm_controller */

    std::string left_robotName = "icubSim"; /* icubSim */
    std::string left_partName = "left_arm"; /* {ARM}_arm */
    std::string left_local = "/left_arm_controller"; /* /{ARM}_arm_controller */

    bool single_mode = true;
    
    if (robotName=="")
    {
        std::cout << "Failed loading config file!" << std::endl;
        return -1;
    }

    /* __________________________________________ YARP NETWORKING __________________________________________ */

    /* configuring in and out ports */
    /* where to connect */
    std::string right_remotePorts="/";
    right_remotePorts+=right_robotName;
    right_remotePorts+="/";
    right_remotePorts+=right_partName;

    /* name of local port */
    std::string right_localPorts = "/right_arm_controller"; /* /{ARM}_arm_controller */

    /* connect to simulated control board */
    Property options;
    options.put("device", "remote_controlboard");
    options.put("local", right_localPorts.c_str());   
    options.put("remote", right_remotePorts.c_str()); 

    /* create a device */
    PolyDriver right_robotDevice(options);
    if (!right_robotDevice.isValid()) {
        yError("(right) Device not available.\n");
        return 0;
    }

    /* create interfaces */
    IControlMode *right_controlMode;
    ITorqueControl *right_torqueControl;
    IEncoders *right_encoders;
    IAxisInfo *right_axInfo;

    /* check if interfaces are available */
    bool ok;
    ok = robotDevice.view(right_controlMode);
    ok = ok && robotDevice.view(right_torqueControl);
    ok = ok && robotDevice.view(right_encoders);
    ok = ok && robotDevice.view(right_axInfo);
    
    if (!ok) {
        yError("Problems acquiring interfaces\n");
        return 0;
    }

    /* __________________________________________ ARM CONTROL START __________________________________________  */
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

    /* nmpc class init */
    dibiman::centralizedNMPC manip_controller(
        urdf_path,
        list_of_joints,
        end_effector_frame_names,
        ids);
  
    /* ______________ Exposing the created Models and Datas ______________ */

    pinocchio::Model & right_model = manip_controller.armList[0].model;
    pinocchio::Data & right_data = manip_controller.armList[0].data;
    pinocchio::FrameIndex & right_hand_id = manip_controller.armList[0].hand_id;

    pinocchio::Model & left_model = manip_controller.armList[1].model;
    pinocchio::Data & left_data = manip_controller.armList[1].data;
    pinocchio::FrameIndex & left_hand_id = manip_controller.armList[1].hand_id;


    /* ______________ Initial velocity and position vectors ______________ */

    Eigen::VectorXd qi_r = pinocchio::neutral(right_model);
    Eigen::VectorXd vi_r = Eigen::VectorXd::Zero(right_model.nv);

    pinocchio::forwardKinematics(right_model, right_data, qi_r);
    pinocchio::updateFramePlacements(right_model, right_data);

    Eigen::VectorXd qi_l = pinocchio::neutral(left_model);
    Eigen::VectorXd vi_l = Eigen::VectorXd::Zero(left_model.nv);

    pinocchio::forwardKinematics(left_model, left_data, qi_l);
    pinocchio::updateFramePlacements(left_model, left_data);

    /* ______________ Setting Up reference Frames ______________ */

    const Eigen::VectorXd right_arm_pos = right_data.oMf[right_hand_id].translation();

    const Eigen::VectorXd left_arm_pos = left_data.oMf[left_hand_id].translation();

    Eigen::VectorXd object_pos = .5f * (right_arm_pos + left_arm_pos);
    pinocchio::SE3 object_frame(Eigen::Matrix3d::Identity(), object_pos);

    /* ______________ Right Hand ______________ */

    const int right_last_joint = right_model.njoints - 1;
    pinocchio::SE3 right_hand_to_object_transform = right_data.oMi[right_last_joint].inverse() * object_frame;

    pinocchio::Frame right_hand_to_object_frame(
            "r_object",
            right_model.frames[right_hand_id].parentJoint,
            right_hand_id,
            right_hand_to_object_transform,
            pinocchio::FrameType::OP_FRAME
            );
            
    right_model.addFrame(right_hand_to_object_frame);
    right_data = pinocchio::Data(right_model);
    pinocchio::forwardKinematics(right_model, right_data, qi_r);
    pinocchio::updateFramePlacements(right_model, right_data);
    const pinocchio::FrameIndex right_object_id = right_model.getFrameId("r_object");

    /* ______________ Left Hand ______________ */

    const int left_last_joint = left_model.njoints - 1;
    pinocchio::SE3 left_hand_to_object_transform = left_data.oMi[left_last_joint].inverse() * object_frame;

    pinocchio::Frame left_hand_to_object_frame(
            "l_object",
            left_model.frames[left_hand_id].parentJoint,
            left_hand_id,
            left_hand_to_object_transform,
            pinocchio::FrameType::OP_FRAME
            );
            
    left_model.addFrame(left_hand_to_object_frame);
    left_data = pinocchio::Data(left_model);
    pinocchio::forwardKinematics(left_model, left_data, qi_l);
    pinocchio::updateFramePlacements(left_model, left_data);
    const pinocchio::FrameIndex left_object_id = left_model.getFrameId("l_object");

    /* ______________ Object Based Reference Creation ______________ */

    Eigen::VectorXd object_pos_ref = right_data.oMf[right_object_id].translation()
        + Eigen::Vector3d(0.02, 0.03, 0.06);

    Eigen::Vector3d rpy(0.0, M_PI/4.0, 0.0);
    Eigen::Matrix3d R_delta = pinocchio::rpy::rpyToMatrix(rpy);

    Eigen::Matrix3d object_Rot_ref = right_data.oMf[right_object_id].rotation() * R_delta;

    pinocchio::SE3 object_transform_ref(object_Rot_ref, object_pos_ref);

    /* ______________ Inverse Kinematics ______________ */

    const size_t n_dim = joint_list.size();
    double* q_r_ref_arr = new double[n_dim];
    Eigen::Map<Eigen::VectorXd> q_r_ref_vec(q_r_ref_arr, n_dim);
    double* v_r_ref_arr = new double[n_dim];
    Eigen::Map<Eigen::VectorXd> v_r_ref_vec(v_r_ref_arr, n_dim);
    v_r_ref_vec.setZero();

    q_r_ref_vec = manip_controller.inverseKinematics(
            right_model,
            right_data,
            qi_r,
            object_transform_ref,
            right_object_id,
            "full"
            );

    std::cout << q_r_ref_vec << std::endl;

    double* u_r_ref_arr = new double[n_dim];
    Eigen::Map<Eigen::VectorXd> u_r_ref_vec(u_r_ref_arr, n_dim);
    u_r_ref_vec = pinocchio::rnea(right_model, right_data, q_r_ref_vec, v_r_ref_vec, v_r_ref_vec);

    double* q_l_ref_arr = new double[n_dim];
    Eigen::Map<Eigen::VectorXd> q_l_ref_vec(q_l_ref_arr, n_dim);
    double* v_l_ref_arr = new double[n_dim];
    Eigen::Map<Eigen::VectorXd> v_l_ref_vec(v_l_ref_arr, n_dim);
    v_l_ref_vec.setZero();

    q_l_ref_vec = manip_controller.inverseKinematics(
            left_model,
            left_data,
            qi_l,
            object_transform_ref,
            left_object_id,
            "full"
            );

    double* u_l_ref_arr = new double[n_dim];
    Eigen::Map<Eigen::VectorXd> u_l_ref_vec(u_l_ref_arr, n_dim);
    u_r_ref_vec = pinocchio::rnea(left_model, left_data, q_l_ref_vec, v_l_ref_vec, v_l_ref_vec);

    /* Right Arm */
    double* q_r_arr = new double[n_dim]; /* joint positions */
    Eigen::Map<Eigen::VectorXd> q_r_vec(q_r_arr, n_dim);
    q_r_vec.setZero();

    double* v_r_arr = new double[n_dim]; /* joint velocities */
    Eigen::Map<Eigen::VectorXd> v_r_vec(v_r_arr, n_dim);
    v_r_vec.setZero();

    /* Left Arm */
    double* q_l_arr = new double[n_dim]; /* joint positions */
    Eigen::Map<Eigen::VectorXd> q_l_vec(q_l_arr, n_dim);
    q_l_vec.setZero();

    double* v_l_arr = new double[n_dim]; /* joint velocities */
    Eigen::Map<Eigen::VectorXd> v_l_vec(v_l_arr, n_dim);
    v_l_vec.setZero();

    /* ____________ Solver Structs Init _________________ */

    dibiman::armTargetParams right_params(q_r_vec.size() + v_r_vec.size(),
            u_r_ref_vec.size());

    right_params.x0 << q_r_vec, v_r_vec;
    right_params.xref << q_r_ref_vec, v_r_ref_vec;
    right_params.uref = u_r_ref_vec;

    dibiman::armTargetParams left_params(q_l_vec.size() + v_l_vec.size(),
            u_l_ref_vec.size());

    left_params.x0 << q_l_vec, v_l_vec;
    left_params.xref << q_l_ref_vec, v_l_ref_vec;
    left_params.uref = u_l_ref_vec;

    std::map<std::string, dibiman::armTargetParams> bimanual_solve_params;
    bimanual_solve_params[ids[0]] = right_params;
    bimanual_solve_params[ids[1]] = left_params;

    std::map<std::string, casadi::DM> u_star = manip_controller.solve(bimanual_solve_params);

    double joint_idx_arr[] = {}
    for (int i = 0; i < 100; i++) {

        for (int j = 0; j < n_dim; j++) {
          right_encoders->getEncoder(idx_joints[j], &q_r_arr[j]);
          q_r_arr[j] = (M_PI/180) * q_r_arr[j];
        }
        // pinocchio::forwardKinematics(reduced_model, data, q_sens_Vec);  
        // pinocchio::updateFramePlacements(reduced_model, data);  
        // std::cout << data.oMf[handID] << std::endl;

        /* send torque commands */
        // ok = torqueControl->setRefTorques(joints, idx_joints, tau);

        /* write to file */
        // logData(file, tau_v, q_meas_v, joints);

        /* send info to other node(arm) */
        
        yarp::os::Time::delay(0.05);
    }
    /* cleanup, effectively useless because at the moment I ctrl+c from while */
    /* later can put this into 'graceful' exit with interrupt ... */

    right_robotDevice.close();
    
    return 0;
}
