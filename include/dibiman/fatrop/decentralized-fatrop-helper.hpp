#pragma once

#include "dibiman/utils/utils.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"

namespace dibiman {

struct DecentralizedFatropHelper
{
    static constexpr int twist_t = 6;
    fatrop::Scalar _dt;
    fatrop::Index _K;

    int nx;
    int nu;

    int nq;
    int nv;

    pinocchio::FrameIndex _ee_frame_idx;
    pinocchio::Model _model;
    pinocchio::Data _data;

    std::unique_ptr<pinocchio::ModelPool> _model_pool;
    std::vector<SpatialTwistDerivatives> _twist_derivatives_pool;

    std::vector<pinocchio::Motion> _twist_pool;
    //std::vector<RNEASecondOrderDerivatives> _rnea_derivatives_pool; unused as it takes too long

    Eigen::VectorXd q_ref;
    Eigen::VectorXd u_ref;

    Eigen::VectorXd x0;
    //Eigen::VectorXd v_0;

    /* written into outside initializer */
    OcpCosts _q;

    void initData(
            const pinocchio::Model & model, 
            const std::string & ee_frame_str,
            const int & K,
            const int & n_threads)
    {
        _model = model;
        _data = pinocchio::Data(_model);
        _model_pool = std::make_unique<pinocchio::ModelPool>(_model, n_threads);

        nq = _model.nq;
        nv = _model.nv;

        nx = nq + nv;
        nu = nv + _model.njoints - 1;

        _ee_frame_idx = _model.getFrameId(ee_frame_str);

        for (int i = 0; i < n_threads; ++i)
        {
            _twist_derivatives_pool.emplace_back(twist_t, nv);
            _twist_pool.emplace_back();
        }

        q_ref(nq);
        q_ref.setZero();
        u_ref(nv);
        u_ref.setZero();

        x_0(nx);
        // v_0(nv);

        _q.x_k = 1;
        _q.u_k = 1;
        _q.x_K = 10;

        _dt = 0.05;
        _K = K;
    }
    
    // DecentralizedFatropHelper(const pinocchio::Model & model, const std::string & ee_frame_str, const int & n_threads) :
    //     initData(model, ee_frame_str, n_threads)
    // {}

    DecentralizedFatropHelper() = default;

};
}

