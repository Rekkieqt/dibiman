/* cpp libraries */
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <list>
#include <algorithm>
#include <filesystem>
#include <stdio.h>

/* Pinocchio libraries */
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

/* Casadi and Eigen */
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

/* NMPC libraries */
#include "dibiman/nmpc/centralizedNMPC.hpp"

using namespace pinocchio;

int main(int argc, char **argv)
{
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
        ids
        );
  

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
    const pinocchio::FrameIndex left_object_id = left_model.getFrameId("l_object");

    /* ______________ Object Based Reference Creation ______________ */

    Eigen::VectorXd object_pos_ref = right_data.oMf[left_object_id].translation()
        + Eigen::Vector3d(0.02, 0.03, 0.06);

    Eigen::Vector3d rpy(0.0, M_PI/4.0, 0.0);
    Eigen::Matrix3d R_delta = pinocchio::rpy::rpyToMatrix(rpy);

    Eigen::Matrix3d object_Rot_ref = right_data.oMf[right_object_id].rotation() * R_delta;

    pinocchio::SE3 object_transform_ref(object_Rot_ref, object_pos_ref);

    /* ______________ Inverse Kinematics ______________ */

    Eigen::VectorXd qf_r = manip_controller.inverseKinematics(
            right_model,
            right_data,
            qi_r,
            object_transform_ref,
            right_object_id,
            "full"
            );

    Eigen::VectorXd qf_l = manip_controller.inverseKinematics(
            left_model,
            left_data,
            qi_l,
            object_transform_ref,
            left_object_id,
            "full"
            );

    /* Controlled joints */
    const size_t n_dim = joint_list.size();
    // double* q_sens = new double[joints]; /* joint positions */
    // double* v_sens = new double[joints]; /* joint velocities */
    // double* a_sens = new double[joints]; /* joint accelerations */
    // double* u_sens = new double[joints]; /* tau -> torque at the joints */ 

    /* dynamic link between double arrays and eigen arrays */
    // Eigen::Map<Eigen::VectorXd> q_sens_Vec(q_sens, joints);
    // Eigen::Map<Eigen::VectorXd> qd_meas_v(qd_all, joints);
    // Eigen::Map<Eigen::VectorXd> qdd_meas_v(qdd_meas, joints);
    // Eigen::Map<Eigen::VectorXd> tau_v(tau, joints);
    // Eigen::Map<Eigen::VectorXd> tau_meas_v(tau_meas, joints);

    for (int i = 0; i < 2; i++) {

        // pinocchio::forwardKinematics(reduced_model, data, q_sens_Vec);  
        // pinocchio::updateFramePlacements(reduced_model, data);  
        // std::cout << data.oMf[handID] << std::endl;

    }
    
    return 0;
}
