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
#include <math.h>

/* Pinocchio libraries */
#include "pinocchio/spatial/explog.hpp"

#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "pinocchio/parsers/urdf.hpp"

/* Casadi and Eigen */
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

/* NMPC libraries */
#include "dibiman/nmpc/baseNMPC.hpp"
#include "dibiman/nmpc/centralizedNMPC.hpp"

using namespace pinocchio;

template<typename T>
bool is_in_vector(const std::vector<T> & vector, const T & elt) {
  return vector.end() != std::find(vector.begin(), vector.end(), elt);
}

int main(int argc, char **argv)
{
    /* +++++++++++ ARM CONTROL START ++++++++++++++*/
    /* arm joints */
    const std::vector<std::string> joint_list = {"shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_prosup", "wrist_pitch", "wrist_yaw"};
    const std::vector<std::string> ids = {"right_icub_arm", "left_icub_arm"};
    const std::vector<std::string> end_effector_frames = {"r_hand", "l_hand"};
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
        end_effector_frames,
        ids
        );
  
    /* remote controller */
    // int idx_joints[] = {0, 1, 2, 3, 4, 5, 6};
    // int joints = 6;

    /* Controlled joints */
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
    /* cleanup, effectively useless because at the moment I ctrl+c from while */
    /* later can put this into 'graceful' exit with interrupt ... */
    
    return 0;
}
