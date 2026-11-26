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
#include "dibiman/nmpc/manipulator-nmpc.hpp"
#include "dibiman/utils/robot.hpp"
#include "dibiman/utils/utils.hpp"

#include <fatrop/fatrop.hpp>

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
    const double dt = 0.05;
    omp_set_num_threads(n_threads);
    dibiman::DecentralizedNMPC man(model, ee_frame, horizon, n_threads);
    dibiman::ManipulatorNMPC man_cas(model, ee_frame, horizon);
    dibiman::OcpCosts q_costs; 
    q_costs.x_k = 25;
    q_costs.u_k = .5;
    q_costs.x_K = 100;

    man.set_dt(dt);
    man.setCosts(q_costs);

    man_cas.set_dt(dt);
    man_cas.setCosts(q_costs);
    man_cas.create();
  
    /* ______________ Initial velocity and position vectors ______________ */

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

    /* ______________ Inverse Kinematics ______________ */

    /*
    Eigen::VectorXd q_ref = dibiman::inverseKinematics(
            model,
            data, 
            q, 
            r_H_ref,
            ee_idx);
            */

    Eigen::VectorXd qref = pinocchio::randomConfiguration(model);

    man.setRef(qref);
    man.setTraj(q, qref, 10);

    man_cas.setRef(qref);

    /* ____________ Simulation _________________ */

    const int T = 12;
    int N = (int)(T/dt);
    pinocchio::Motion s_twist;

    Eigen::VectorXd x_next(x.size());
    x << q, v;

    /* _________________ Comparison of returned vs expected _________________ */

    std::vector<pinocchio::Motion> twist_ref_list, twist_real_list;
    for (int k = 0; k < horizon; ++k)
    {
        pinocchio::Motion ref_twist = man.traj->s_twist(k * dt);
        twist_ref_list.push_back(ref_twist);
        // std::cout << "ref: " << ref_twist << "\n\n";
    }
    tau = man.solve(x);
    const auto & info = man.ip_data->info();
    for (int k = 0; k < horizon; ++k)
    {
        assert(man.ip_data);
        assert(horizon <= info.dims.K);
        const fatrop::VecRealView & xsol = man.ip_data->current_iterate().primal_x();
        assert(xsol.m() == info.number_of_primal_variables);
        const fatrop::Scalar *xs = xsol.data() + info.offsets_primal_x[k];
        Eigen::VectorXd _xsol(2 * model.nv);
        std::memcpy(_xsol.data(), xs, (size_t)man.ocp->nx * sizeof(fatrop::Scalar));
        // for (int i = 0; i < (2*model.nv); ++i) _xsol[i] = xs[i];
        pinocchio::forwardKinematics(model, data, _xsol.head(model.nq), _xsol.tail(model.nv));
        pinocchio::Motion real_twist = pinocchio::getFrameVelocity(model, data, ee_idx, pinocchio::WORLD);
        twist_real_list.push_back(real_twist);
        // std::cout << "real: " << real_twist << "\n\n";
    }

    for (int k = 0; k < horizon; ++k)
    {
        pinocchio::Motion diff = twist_real_list[k] - twist_real_list[k];
        // std::cout << "diff: " << diff.toVector().norm() << "\n\n";
    }

    /* _________________ NMPC simulation over T time _________________ */
    for (int i = 0; i < 4; i++) {

        tau = man.solve(x);
        x_next = dibiman::descreteForwardDynamics(model, x, tau, dt);
        x = x_next;

        pinocchio::forwardKinematics(model, data, x.head(model.nq), x.tail(model.nv));
        pinocchio::updateFramePlacements(model, data);
        s_twist = getFrameVelocity(model, data, ee_idx, pinocchio::WORLD);
        // std::cout << "Spatial Twist: " << s_twist << "\n\n";
        // std::cout << "Spatial Twist Expected: " << man.traj->s_twist(0) << "\n\n";
        // std::cout << "Spatial Twist Diff Norm: " << (s_twist - man.traj->s_twist(0)).toVector().norm() << "\n\n";

    }
    // std::cout << "Return flag: " << int(ret) << std::endl;
    std::cout << man.ip_data->timing_statistics() << std::endl;
    std::cout << "Final x: " << x.transpose() << "\n\n";
    std::cout << "Desired q: " << qref.transpose() << "\n\n";
    return 0;
}

