#include <limits>
#include <memory>
#include <omp.h>
#include <fatrop/fatrop.hpp>

#include "dibiman/utils/robot.hpp"
#include "dibiman/utils/utils.hpp"
#include "dibiman/utils/trajectory-generation.hpp"
#include "dibiman/fatrop/static-decentralized-ocp.hpp"
#include "dibiman/fatrop/decentralized-fatrop-helper.hpp"

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"

#include "pinocchio/spatial.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

using namespace fatrop;
using namespace Eigen;

namespace fatrop {

typedef OcpAbstractTpl<dibiman::DecentralizedManipulatorStatic> DecentralizedOcp;

DecentralizedOcp::OcpAbstractTpl(
        const pinocchio::Model & model, 
        const std::string & ee_frame_str, 
        const int & K, 
        const int & n_threads,
        const bool & trajectory)
{
    initData(model, ee_frame_str, K, n_threads);
    _trajectory_control = trajectory;
}

Index DecentralizedOcp::get_nx(const Index k) const { return nx; }

Index DecentralizedOcp::get_nu(const Index k) const
{
    if (k == _K - 1)
    {
        return 0;
    }
    else
    {
        return nu;
    }
}

Index DecentralizedOcp::get_ng(const Index k) const
{
    int ng_size = nv;
    if (_trajectory_control) ng_size += twist_t;
    if (k == 0)
    {
        return ng_size += nx; // twist_t + nx + nv; 
    }
    else if (k == _K - 1)
    {
        return 0;
    }
    else
    {
        // if (trajectory_control)
        return ng_size; // nv + twist_t;
    }
};

Index DecentralizedOcp::get_ng_ineq(const Index k) const { return 0; };
Index DecentralizedOcp::get_horizon_length() const { return _K; };
Index DecentralizedOcp::eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
                MAT *res, const Index k)
{
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);

    // B.T
    blasfeo_diare_wrap(nq, 1.f, res, 0, nq); 
    // A.T
    blasfeo_diare_wrap(nq, 1.f, res, nx, 0); 
    blasfeo_diare_wrap(nq, _dt, res, nx + nq, 0); 
    blasfeo_diare_wrap(nq, 1.f, res, nx + nq, nq); 
    return 0;
}
Index DecentralizedOcp::eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                  const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k,
                  MAT *res, const Index k)
{
    // set zero
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    const Scalar s = objective_scale[0];

    if (k == _K - 1)
    {                              
        blasfeo_diare_wrap(nx, s * 2 * _qcost.x_K, res, 0, 0);
        // return 0;
    }                              
    else
    {                              
        blasfeo_diare_wrap(nu, s * 2 * _qcost.u_k, res, 0, 0);
        blasfeo_diare_wrap(nx, s * 2 * _qcost.x_k, res, nu, nu);
    }

    /* ==== Full RNEA Hessian Implementation ====
    const int thread_id = omp_get_thread_num();
    const pinocchio::Model & _model = _model_pool->getModels()[(size_t)thread_id];
    pinocchio::Data & _data = _model_pool->getDatas()[(size_t)thread_id];
    dibiman::RNEASecondOrderDerivatives & _m = _rnea_derivatives_pool[(size_t)thread_id];
    _m.setZero();

    // Put RNEA and Jacobian Derivatives here
    Map<const VectorXd> _q  (states_k, nq);
    Map<const VectorXd> _v  (states_k + nq, nq);
    Map<const VectorXd> _a  (inputs_k, nq);
    TensorMap<const Tensor<double, 1>> _lam_eq  (lam_eq_k, nq);

    ComputeRNEASecondOrderDerivatives(_model, _data, _q, _v, _a);

    // Tensor contraction
    const Eigen::array<Eigen::IndexPair<int>, 1> dims = { Eigen::IndexPair<int>(0, 0) };
    _m.qq = _lam_eq.contract(_data.d2tau_dqdq, dims);
    _m.vv = _lam_eq.contract(_data.d2tau_dvdv, dims);
    _m.qv = _lam_eq.contract(_data.d2tau_dqdv, dims);
    _m.aq = _lam_eq.contract(_data.d2tau_dadq, dims);

    Map<MatrixXd> _qq_mat(_m.qq.data(), nq, nq);
    Map<MatrixXd> _vv_mat(_m.vv.data(), nq, nq);
    Map<MatrixXd> _qv_mat(_m.qv.data(), nq, nq);
    Map<MatrixXd> _aq_mat(_m.aq.data(), nq, nq);

    _qq_mat.diagonal().array() += s * 2 * _qcost.x_k;
    _vv_mat.diagonal().array() += s * 2 * _qcost.x_k;

    blasfeo_pack_dmat(nq, nq,
            _qq_mat.data(), _qq_mat.outerStride(), 
            res, nx, nx);
    blasfeo_pack_dmat(nq, nq,
            _vv_mat.data(), _vv_mat.outerStride(), 
            res, nx + nq, nx + nq);
    blasfeo_pack_dmat(nq, nq,
            _qv_mat.data(), _qv_mat.outerStride(), 
            res, nx, nx + nq);
    blasfeo_pack_dmat(nq, nq,
            _aq_mat.data(), _aq_mat.outerStride(), 
            res, 0, nx);

    blasfeo_pack_tran_dmat(nq, nq,
            _qv_mat.data(), _qv_mat.outerStride(), 
            res, nx + nq, nx);
    blasfeo_pack_tran_dmat(nq, nq,
            _aq_mat.data(), _aq_mat.outerStride(), 
            res, nx, 0);
            */
    return 0;
};
Index DecentralizedOcp::eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
{
    if (k == _K - 1) return 0;

    /* RNEA constraint
     * h(q, v, a, tau) = rnea(q, v, a) - tau = 0
     * dh/da, dh/dtau, dh/dq, dh/dv = dh/d(uk, xk)
     * dh/dtau = I_nq 
     * Ggt = [data.M, I, data.dtau_dq, data.dtau_dv]
     * 
     *
     * Jacobian constraint
     * h(q, v, a, tau) = J(q) * v - v_ref (const w.r.t. (q, v))
     * dh/da, dh/dtau, dh/dq, dh/da = dh/d(uk, xk)
     * [0, 0, dh/dq, J(q)] = dh/d(uk, xk)
     * dh/da = 0
     * dh/dtau = 0
     * dh/duk = 0
     */

    const int thread_id = omp_get_thread_num();
    const pinocchio::Model & _model = _model_pool->getModels()[(size_t)thread_id];
    pinocchio::Data & _data = _model_pool->getDatas()[(size_t)thread_id];

    // Put RNEA and Jacobian Derivatives here
    Map<const VectorXd> _q  (states_k, nq);
    Map<const VectorXd> _v  (states_k + nq, nq);
    Map<const VectorXd> _a  (inputs_k, nq);

    pinocchio::computeRNEADerivatives(_model, _data, _q, _v, _a);

    /// \note The results are stored in data.dtau_dq, data.dtau_dv and data.M which respectively
    /// correspond
    ///          to the partial derivatives of the joint torque vector with respect to the joint
    ///          configuration, velocity and acceleration. As for pinocchio::crba, only the upper
    ///          triangular part of data.M is filled.

    // if (trajectory_control)

    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    pinocchio::make_symmetric(_data.M);

    blasfeo_pack_dmat(_data.M.rows(), _data.M.cols(), _data.M.data(), _data.M.outerStride(), res, 0, 0);
    blasfeo_diare_wrap(nq, -1.f, res, nq, 0);
    blasfeo_pack_tran_dmat(_data.dtau_dq.rows(), _data.dtau_dq.cols(), _data.dtau_dq.data(), _data.dtau_dq.outerStride(), res, nx, 0);
    blasfeo_pack_tran_dmat(_data.dtau_dv.rows(), _data.dtau_dv.cols(), _data.dtau_dv.data(), _data.dtau_dv.outerStride(), res, nx + nq, 0);

    int x0_cols_offset = nq;
    if (_trajectory_control)
    {
        dibiman::SpatialTwistDerivatives & _v_data = _twist_derivatives_pool[(size_t)thread_id];
        _v_data.setZero();

        /* Needs to be called after rnea */
        pinocchio::getFrameVelocityDerivatives(_model, _data, _ee_frame_idx, pinocchio::WORLD, _v_data.dq, _v_data.dv);

        blasfeo_pack_tran_dmat(_v_data.dq.rows(), _v_data.dq.cols(), _v_data.dq.data(), _v_data.dq.outerStride(), res, nx, nq);
        blasfeo_pack_tran_dmat(_v_data.dv.rows(), _v_data.dv.cols(), _v_data.dv.data(), _v_data.dv.outerStride(), res, nx + nv, nq);
        x0_cols_offset += twist_t;
    }

    if (k == 0)
    {
        /* _trajectory_control -> offset on the x[0] = x_init equality */
        blasfeo_diare_wrap(nx, 1.f, res, nu, x0_cols_offset);
    }
    return 0;
}
Index DecentralizedOcp::eval_b(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
             Scalar *res, const Index k)
{
    for (int i = 0; i < nq; ++i)
    {
        res[i] = -states_kp1[i] + states_k[i] + _dt * states_k[i + nq];
        res[i + nq] = -states_kp1[i + nq] + states_k[i + nq] + inputs_k[i];
    }
    return 0;
}
Index DecentralizedOcp::eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
{
    const int thread_id = omp_get_thread_num();
    const pinocchio::Model & _model = _model_pool->getModels()[(size_t)thread_id];
    pinocchio::Data & _data = _model_pool->getDatas()[(size_t)thread_id];
    pinocchio::Motion & _twist = _twist_pool[(size_t)thread_id];

    fatrop::Index k0_offset_q = nq;
    fatrop::Index k0_offset_v = nx;

    if (k == _K - 1) { return 0; }
    else
    {
        Map<const VectorXd> _q  (states_k, nq);
        Map<const VectorXd> _v  (states_k + nq, nq);
        Map<const VectorXd> _a  (inputs_k, nq);

        pinocchio::rnea(_model, _data, _q, _v, _a);

        for (int i = 0; i < nq; ++i)
        {
            res[i] = _data.tau[i] - inputs_k[nq + i];
        }
        if (_trajectory_control) 
        {
            _twist = getFrameVelocity(_model, _data, _ee_frame_idx, pinocchio::WORLD);
            const pinocchio::Motion _twist_err = _twist - _traj->s_twist(_dt * k);
            for (int i = 0; i < twist_t; ++i)
            {
                res[nq + i] = _twist_err.toVector()[i];
                // res[nq + i] = _twist.toVector()[i] - _ref_twist.toVector()[i]; // - _twist_ref[i]; // vel_ref uncreated yet
            }
            k0_offset_q += twist_t;
            k0_offset_v += twist_t;
        }
    }

    if (k == 0)
    {
        for (int i = 0; i < nq; ++i)
        {
            res[k0_offset_q + i] = states_k[i] - x0[i]; // - ... q0
            res[k0_offset_v + i] = states_k[i + nq] - x0[i + nq]; // - ... v0
        }
    }
    return 0;
};
Index DecentralizedOcp::eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
              Scalar *res, const Index k)
{
    const Scalar s = objective_scale[0];

    if (k == _K - 1)
    {
        for (int i = 0; i < nx; ++i)
        {
            const Scalar x_err = states_k[i] - xref[i];
            res[i] = 2 * _qcost.x_K * x_err * s;
        }
    }
    else
    {
        for (int i = 0; i < nx; ++i)
        {
            const Scalar u_err = inputs_k[i] - uref[i];
            const Scalar x_err = states_k[i] - xref[i];

            res[i] = 2 * _qcost.u_k * u_err * s;
            res[nx + i] = 2 * _qcost.x_k * x_err * s;
        }
    }
    return 0;
}
Index DecentralizedOcp::eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
             Scalar *res, const Index k)
{
    *res = 0.0;
    const Scalar s = objective_scale[0];

    if (k == _K - 1)
    {
        for (int i = 0; i < nx; ++i)
        {
            const Scalar err = states_k[i] - xref[i];
            *res+= _qcost.x_K * err * err;
        }
        *res*= s;
    }
    else
    {
        for (int i = 0; i < nx; ++i)
        {
            const Scalar u_err = inputs_k[i] - uref[i];
            const Scalar x_err = states_k[i] - xref[i];
            *res+= _qcost.u_k * u_err * u_err + _qcost.x_k * x_err * x_err;
        }
        *res*= s;
    }
    return 0;
}
Index DecentralizedOcp::eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
{
    return 0;
};
Index DecentralizedOcp::eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
{
    return 0;
};
Index DecentralizedOcp::get_bounds(Scalar *lower, Scalar *upper, const Index k) const
{
    return 0;
}

Index DecentralizedOcp::get_initial_xk(Scalar *xk, const Index k) const
{
    if (_warm) 
    {
        const Eigen::VectorXd & _w_xk = _warm_x[k];
        for (int i = 0; i < nx; ++i) xk[i] = _w_xk[i];
        // std::copy_n (...) - >xk = _w_xk.data();
        return 0;
    }
    for (int i = 0; i < nx; ++i) xk[i] = x0[i];
    return 0;
};
Index DecentralizedOcp::get_initial_uk(Scalar *uk, const Index k) const
{
    if (k == _K - 1) return 0;
    if (_warm) 
    {
        const Eigen::VectorXd & _w_uk = _warm_u[k];
        for (int i = 0; i < nx; ++i) uk[i] = _w_uk[i];
        // std::copy_n (...) -> uk = _w_uk.data();
        return 0;
    }
    // Eigen::VectorXd & q0 = x0.head(nq);
    // Eigen::VectorXd & v0 = x0.tail(nv);
    // pinocchio::rnea(_model, _data, q0, v0, v0);
    for (int i = 0; i < nv; ++i) uk[nv + i] = uref[nv + i];
    return 0;
};
}
