#pragma once

#include "dibiman/utils/utils.hpp"
#include "dibiman/utils/trajectory-generation.hpp"
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
    std::vector<RNEASecondOrderDerivatives> _rnea_derivatives_pool; // unused as it takes too long

    Eigen::VectorXd xref;
    Eigen::VectorXd uref;

    Eigen::VectorXd x0;
    //Eigen::VectorXd v_0;

    /* written into outside initializer */
    OcpCosts _qcost;
    bool _warm = false;

    std::vector<Eigen::VectorXd> _warm_u;
    std::vector<Eigen::VectorXd> _warm_x;

    bool _trajectory_control = false;
    std::shared_ptr<TrajectoryGenerator> _traj;

    void initData(
            const pinocchio::Model & model, 
            const std::string & ee_frame_str,
            const int & K,
            const int & n_threads = -1)
    {
        _model = model;
        _data = pinocchio::Data(_model);

        nq = _model.nq;
        nv = _model.nv;

        nx = nq + nv;
        nu = 2 * nv; // u = [a, tau]^T

        xref.resize(nx);
        xref.setZero();
        uref.resize(nu); // only tau is used
        uref.setZero();
        x0.resize(nx);
        x0.setZero();

        _qcost.x_k = 1;
        _qcost.u_k = 1;
        _qcost.x_K = 10;

        _dt = 0.05;
        _K = K;

        _warm_u.reserve(_K - 1);
        _warm_x.reserve(_K);

        for (int k = 0; k < _K; ++k)
        {
            _warm_x.emplace_back(nx);
            if (k < _K - 1) _warm_u.emplace_back(nu);
        }

        if (n_threads == -1) return;

        _model_pool = std::make_unique<pinocchio::ModelPool>(_model, n_threads);
        _ee_frame_idx = _model.getFrameId(ee_frame_str);

        for (int i = 0; i < n_threads; ++i)
        {
            _twist_derivatives_pool.emplace_back(nv);
            _rnea_derivatives_pool.emplace_back(nv);
            _twist_pool.emplace_back();
        }
    }

    // DecentralizedFatropHelper(const pinocchio::Model & model, const std::string & ee_frame_str, const int & n_threads) :
    //     initData(model, ee_frame_str, n_threads)
    // {}

    DecentralizedFatropHelper() = default;

};
}

