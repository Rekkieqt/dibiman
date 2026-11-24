#pragma once

#ifndef __dibiman_utils_utils__
#define __dibiman_utils_utils__

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/eigen-common.hpp"
#include <fatrop/fatrop.hpp>
#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace dibiman {

    struct ManipulatorParameters {
        std::string model_path;
        std::string end_effector_frame_name;
        std::vector<std::string> reduced_model_joints;
    };

    struct ObjectParameters {
        std::string model_path;
        // shape info ...
    };

    struct OcpCosts {
        /* Running state and control cost */
        fatrop::Scalar x_k;

        fatrop::Scalar q_k;
        fatrop::Scalar v_k;
        fatrop::Scalar u_k;

        fatrop::Scalar x_K; /* Terminal state cost */

        fatrop::Scalar q_K; /* Terminal state cost */
        fatrop::Scalar v_K; /* Terminal state cost */

        OcpCosts() :
            x_k(1),
            q_k(1),
            v_k(1),
            u_k(1),
            x_K(10),
            v_K(10)
        {}

    };

    struct SpatialTwistDerivatives {
        pinocchio::Data::Matrix6x dq;
        pinocchio::Data::Matrix6x dv;

        SpatialTwistDerivatives() = default;

        SpatialTwistDerivatives(const int & nv) :
            dq(6, nv),
            dv(6, nv)
        {}

        void setZero(void)
        {
            dq.setZero();
            dv.setZero();
        }
    };

    struct RNEASecondOrderDerivatives {
        Eigen::Tensor<double, 2> qq;
        Eigen::Tensor<double, 2> qv;
        Eigen::Tensor<double, 2> vv;
        Eigen::Tensor<double, 2> aq;

        RNEASecondOrderDerivatives() = default;

        RNEASecondOrderDerivatives(const int nv) :
            qq(nv, nv),
            qv(nv, nv),
            vv(nv, nv),
            aq(nv, nv)
        {}

        void setZero(void)
        {
            qq.setZero();
            qv.setZero();
            vv.setZero();
            aq.setZero();
        }
    };

    struct armTargetParams {
        Eigen::VectorXd x0;
        Eigen::VectorXd xref;
        Eigen::VectorXd uref;

        armTargetParams() = default;

        armTargetParams(int nv)
            : x0(nv + nv),
            xref(nv + nv),
            uref(nv)

        {
            // x0.setZero();
            // xref.setZero();
            // uref.setZero();
        }

        // double* q0 = nullptr;
        // double* v0 = nullptr;
        // double* uref = nullptr;
        // double* qref = nullptr;
        // double* vref = nullptr;
        
        // size_t n_dim = 0; /* keep track of alloc size */

        // ``~armTargetParams() {
        // ``    delete[] q0, v0, uref, qref, vref;
        // ``}
    };

    inline casadi::DM eigenToDM(const Eigen::VectorXd& vec)
    {
        casadi::DM dm = casadi::DM::zeros(vec.size());
        std::memcpy(dm.ptr(), vec.data(), sizeof(double) * vec.size());
        return dm;
    }
}

/*

void logData(std::fstream & logfile, Eigen::Ref<Eigen::VectorXd> tau, Eigen::Ref<Eigen::VectorXd> qread, int size) {
  for (int i=0; i < size; ++i) {
    logfile << qread[i] << ",";
  }
  for (int i=0; i < size; ++i) {
    logfile << tau[i] << ",";
  }
  logfile << std::endl;
}

void logHeader(std::fstream & logfile, int size) {
  for (int i=0; i < size; ++i) {
    logfile << "joint_meas" << i << ",";
  }
  for (int i=0; i < size; ++i) {
    logfile << "torque " << i << ",";
  }
  logfile << std::endl;
}

void logRef(std::fstream & reffile, Eigen::VectorXd& qref, int size) {
  for (int i=0; i < size; ++i) {
    reffile << "joint_ref" << i << ",";
  }
  reffile << std::endl;
  for (int i=0; i < size; ++i) {
    reffile << qref[i] << ",";
  }
  reffile << std::endl;
}

*/
#endif //__dibiman_utils_utils__
