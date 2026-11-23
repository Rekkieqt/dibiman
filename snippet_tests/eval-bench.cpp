#include <iostream>
#include <chrono>
#include <filesystem>
#include "dibiman/utils/robot.hpp"

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/spatial.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

using namespace Eigen;
    

int
main()
{
    const std::string modelpath = static_cast<std::string>(std::filesystem::current_path()) + "/../../conf/model.urdf";
    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};

    pinocchio::Model _model = dibiman::returnReducedModel(modelpath, joint_list);
    pinocchio::Data _data = pinocchio::Data(_model);

    VectorXd _q = pinocchio::neutral(_model);
    VectorXd _v(VectorXd::Random(_model.nv));
    VectorXd _a(VectorXd::Random(_model.nv));

    pinocchio::FrameIndex _ee_frame_id = _model.getFrameId("r_hand_dh_frame");

    pinocchio::Data::Matrix6x partial_dq(6, _model.nv), partial_dv(6, _model.nv);
    const int test_size = 10;
    std::vector<std::chrono::microseconds::rep> rnea_list;
    std::vector<std::chrono::microseconds::rep> Jv_list;

    for (int i = 0; i < test_size; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();  
        pinocchio::computeRNEADerivatives(_model, _data, _q, _v, _a);  
        auto t1 = std::chrono::steady_clock::now();  
        pinocchio::getFrameVelocityDerivatives(_model, _data, _ee_frame_id, pinocchio::WORLD, partial_dq, partial_dv);
        auto t2 = std::chrono::steady_clock::now();

        // std::chrono::duration<double, std::chrono::microseconds> rnea_jac_elapsed = (t1 - t0);
        auto rnea_us = std::chrono::duration_cast<std::chrono::microseconds> (t1 - t0).count();
        // std::chrono::duration<double, std::chrono::microseconds> Jv_jac_elapsed = (t2 - t1);
        auto Jv_us = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();

        partial_dv.setZero();
        partial_dq.setZero();
        rnea_list.push_back(rnea_us);
        Jv_list.push_back(Jv_us);
    }

    for (int i = 0; i < test_size; ++i)
    {
        std::cout << "RNEA Jac(" << i << ") :"  << rnea_list[i]<< "\n";
        std::cout << "J(q)v Jac(" << i << ") :" << Jv_list[i] << "\n";
    }

    return 0;
}
