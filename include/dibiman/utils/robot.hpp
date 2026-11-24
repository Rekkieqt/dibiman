#pragma once

#ifndef __dibiman_utils_robot__
#define __dibiman_utils_robot__

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  

#include <casadi/casadi.hpp>

namespace dibiman {

    pinocchio::Model::ConfigVectorType inverseKinematics(
        const pinocchio::Model &,  
        pinocchio::Data &,  
        const pinocchio::Model::ConfigVectorType &,
        const pinocchio::SE3 &,
        const pinocchio::FrameIndex &,
        const std::string & = "full");

    pinocchio::Model returnReducedModel(
        const std::string &,
        const std::vector<std::string> &);

    /* delta time is only used once in practice */
    Eigen::VectorXd descreteForwardDynamics(const pinocchio::Model &, const Eigen::VectorXd &, const Eigen::VectorXd &, const double &);

    casadi::Function descreteABA(const pinocchio::Model &, const double &);

    casadi::Function linearDynamics(const int, const double);

    casadi::Function rneaSymbolic(const pinocchio::Model &);

};
#endif //__dibiman_utils_robot__
