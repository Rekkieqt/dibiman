#pragma once

#include <limits>
#include <memory>
#include <omp.h>
#include <fatrop/fatrop.hpp>
#include "dibiman/fatrop/nlp_ocp.hxx"
#include "dibiman/utils/utils.hpp"

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"

#include "pinocchio/spatial.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

namespace dibiman {

class DecentralizedFatropHelper
{
    protected:
        static constexpr int twist_t = 6;
        fatrop::Scalar _dt;
        fatrop::Index _K;

        int nx;
        int nu;

        int nq;
        int nv;

        fatrop::Scalar _Qx_k;
        fatrop::Scalar _Qu_k;

        fatrop::Scalar _Qx_N;
        fatrop::Scalar _Qv_N;

        pinocchio::FrameIndex _ee_frame_id;
        pinocchio::Model _model;
        pinocchio::Data _data;

        std::unique_ptr<pinocchio::ModelPool> _model_pool;
        std::vector<SpatialTwistDerivatives> _twist_derivatives_pool;

        std::vector<pinocchio::Motion> _twist_pool;
        //std::vector<RNEASecondOrderDerivatives> _rnea_derivatives_pool;
        //
        Eigen::VectorXd q_ref;
        Eigen::VectorXd u_ref;

        void initPinData(const std::string & modelpath, const std::vector<std::string> & list_joints_to_use, const int & n_threads) {
            _model = dibiman::getArmModel(modelpath, list_joints_to_use);
            _data = pinocchio::Data(_model);
            _model_pool = std::make_unique<pinocchio::ModelPool>(_model, n_threads);
            nq = _model.nq;
            nv = _model.nv;
            nx = nq * 2;
            nu = nq * 2;
            // _mat_pool.reserve(n_threads);
            _ee_frame_id = _model.getFrameId("r_hand_dh_frame");
            for (int i = 0; i < n_threads; ++i)
            {
                // _mat_pool.emplace_back(nq);
                _twist_derivatives_pool.emplace_back(twist_t, nv);
                _twist_pool.emplace_back();
            }
        }

        // inline double get_s(const double & t, const double & Tf)
        // {
        //     double x = t / Tf;
        //     double x3 = x * x * x;
        //     return a3 * x3 + a4 * x3 * x + a5 * x3 * x * x;
        // }

        // inline double get_ds(const double & t, const double & Tf)
        // {
        //     double x = t / Tf;
        //     double x2 = x * x;
        //     double res = 3 * a3 * x2 + 4 * a4 * x2 * x + 5 * a5 * x2 * x2;
        //     return res * (1.0)/Tf;
        // }

        // inline pinocchio::Motion twist_ref(const fatrop::Index k)
        // {
        //     double s = get_s(k * dt_, Tf);
        //     double ds = get_ds(k * dt_, Tf);
        //     pinocchio::Motion nu = pinocchio::log6(Xi.actInv(Xf));
        //     pinocchio::SE3 Xs = Xi * pinocchio::exp6(nu * s);
        //     return Xs.act(nu) * ds;
        // }
};
    
}

