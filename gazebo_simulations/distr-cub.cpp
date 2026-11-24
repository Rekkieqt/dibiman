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
#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

/* OpenMP */
#include "omp.h"

/* NMPC libraries */
#include "dibiman/nmpc/distr-nmpc.hpp"
#include "dibiman/utils/robot.hpp"

using namespace pinocchio;

int 
main()
{
    /* __________________________________________ CONTROL START __________________________________________  */

    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};
    const std::vector<std::string> ids = {"right_icub_arm", "left_icub_arm"};
    const std::string ee_frame = "r_hand_dh_frame";
    const std::string model_path = "/home/ubuntu/dibiman/conf/model_FT.urdf";

    pinocchio::Model model = dibiman::returnReducedModel(model_path, joint_list);
    pinocchio::Data data = pinocchio::Data(model);

    const int horizon = 50;
    const int n_threads = 4;
    omp_set_num_threads(n_threads);
    dibiman::DecentralizedNMPC man(model, ee_frame, horizon, n_threads);
    // man.set_dt(0.01);
  
    /* ______________ Initial velocity and position vectors ______________ */

    std::cerr << "Linking yarp q and v to Eigen\n\n";
    constexpr double degToPi = M_PI/180;

    Eigen::VectorXd q(model.nv);
    q.setZero();

    Eigen::VectorXd v(model.nv);
    v.setZero();

    Eigen::VectorXd tau(model.nv);

    Eigen::VectorXd x(model.nv + model.nv);

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    /* ______________ Object grasp movement 1 - hover ______________ */

    const pinocchio::FrameIndex ee_idx = model.getFrameId(ee_frame);
    const Eigen::Vector3d pos_ref = data.oMf[ee_idx].translation() + Eigen::Vector3d(0.0, 0.1, 0.0);

    const Eigen::Vector3d right_rpy(M_PI/2.0f, M_PI, 0);
    const Eigen::Matrix3d r_R = pinocchio::rpy::rpyToMatrix(right_rpy);
    pinocchio::SE3 r_H_ref(r_R, pos_ref);
    std::cout << "Right H ref : " << r_H_ref << std::endl;

    /* ______________ Inverse Kinematics ______________ */

    Eigen::VectorXd q_ref = dibiman::inverseKinematics(
            model,
            data, 
            q, 
            r_H_ref,
            ee_idx);

    man.setRef(q_ref);

    /* ____________ Solver Structs Init _________________ */

    for (int i = 0; i < 10; i++) {

        x << q, v;
        tau = man.solve(x);

        pinocchio::forwardKinematics(model, data, q * degToPi);
        pinocchio::updateFramePlacements(model, data);

    }

    return 0;
}

