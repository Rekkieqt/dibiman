#include <limits>
#include <filesystem>
#include <memory>
#include <omp.h>
#include <fatrop/fatrop.hpp>
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

using namespace fatrop;
using namespace Eigen;

namespace fatrop {

typedef fatrop::OcpAbstractTpl<RobotManipulator> ocp;

ocp::OcpAbstractTpl(const std::string & modelpath, const std::vector<std::string> & joints, const int & n_threads)
{
    initPinData(modelpath, joints, n_threads);
}

Index ocp::get_nx(const Index k) const { return nx; }

Index ocp::get_nu(const Index k) const
{
    if (k == K_ - 1)
    {
        return 0;
    }
    else
    {
        return nu;
    }
}

Index ocp::get_ng(const Index k) const
{
    if (k == 0)
    {
        return vel_size + nx + nv;
    }
    else if (k == K_ - 1)
    {
        return 0;
    }
    else
    {
        return nv + vel_size;
    }
};

Index ocp::get_ng_ineq(const Index k) const { return 0; };
Index ocp::get_horizon_length() const { return K_; };
Index ocp::eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
                MAT *res, const Index k)
{
    // set zero
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);

    // B.T
    blasfeo_diare_wrap(nq, 1.f, res, 0, nq); 
    // A.T
    blasfeo_diare_wrap(nq, 1.f, res, nx, 0); 
    blasfeo_diare_wrap(nq, dt_, res, nx + nq, 0); 
    blasfeo_diare_wrap(nq, 1.f, res, nx + nq, nq); 
    return 0;
}
Index ocp::eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                  const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k,
                  MAT *res, const Index k)
{
    // set zero
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    const Scalar s = objective_scale[0];

    if (k == K_ - 1)
    {                              
        blasfeo_diare_wrap(nx, s * 2 * _Qx_N, res, 0, 0);
        // return 0;
    }                              
    else
    {                              
        blasfeo_diare_wrap(nu, s * 2 * _Qu_k, res, 0, 0);
        blasfeo_diare_wrap(nx, s * 2 * _Qx_k, res, nu, nu);
    }
    return 0;
};
Index ocp::eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
{
    if (k == K_ - 1) return 0;

    /* RNEA constraint
     * h(q, v, a, tau) = rnea(q, v, a) - tau = 0
     * dh/da, dh/dtau, dh/dq, dh/dv = dh/d(uk, xk)
     * dh/dtau = I_nq 
     * Ggt = [data.M, I, data.dtau_dq, data.dtau_dv]
     * 
     *
     * Jacobian constraint
     * h(q, v, a, tau) = J(q) * v - (numerical v_des)
     * dh/da, dh/dtau, dh/dq, dh/da = dh/d(uk, xk)
     * [0, 0, dh/dq, J(q)] = dh/d(uk, xk)
     * dh/da = 0
     * dh/dtau = 0
     * dh/duk = 0
     */

    const int thread_id = omp_get_thread_num();
    const pinocchio::Model & _model = _pool->getModels()[(size_t)thread_id];
    pinocchio::Data & _data = _pool->getDatas()[(size_t)thread_id];
    _frame_velocity_derivatives & _v_data = _vel_partial_pool[(size_t)thread_id];
    _v_data.setZero();

    // Put RNEA and Jacobian Derivatives here
    Map<const VectorXd> _q  (states_k, nq);
    Map<const VectorXd> _v  (states_k + nq, nq);
    Map<const VectorXd> _a  (inputs_k, nq);

    pinocchio::computeRNEADerivatives(_model, _data, _q, _v, _a);

    pinocchio::getFrameVelocityDerivatives(_model, _data, _ee_frame_id, pinocchio::WORLD, _v_data.dq, _v_data.dv);

    /// \note The results are stored in data.dtau_dq, data.dtau_dv and data.M which respectively
    /// correspond
    ///          to the partial derivatives of the joint torque vector with respect to the joint
    ///          configuration, velocity and acceleration. As for pinocchio::crba, only the upper
    ///          triangular part of data.M is filled.

    // set zero
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    pinocchio::make_symmetric(_data.M);

    blasfeo_pack_dmat(_data.M.rows(), _data.M.cols(), _data.M.data(), _data.M.outerStride(), res, 0, 0);
    blasfeo_diare_wrap(nq, -1.f, res, nq, 0);
    blasfeo_pack_tran_dmat(_data.dtau_dq.rows(), _data.dtau_dq.cols(), _data.dtau_dq.data(), _data.dtau_dq.outerStride(), res, nx, 0);
    blasfeo_pack_tran_dmat(_data.dtau_dv.rows(), _data.dtau_dv.cols(), _data.dtau_dv.data(), _data.dtau_dv.outerStride(), res, nx + nq, 0);

    blasfeo_pack_tran_dmat(_v_data.dq.rows(), _v_data.dq.cols(), _v_data.dq.data(), _v_data.dq.outerStride(), res, nx, nq);
    blasfeo_pack_tran_dmat(_v_data.dv.rows(), _v_data.dv.cols(), _v_data.dv.data(), _v_data.dv.outerStride(), res, nx + nv, nq);

    if (k == 0)
    {
        blasfeo_diare_wrap(nx, 1.f, res, nu, nq + vel_size);
    }
    return 0;
}
Index ocp::eval_b(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
             Scalar *res, const Index k)
{
    for (int i = 0; i < nq; ++i)
    {
        res[i] = -states_kp1[i] + states_k[i] + dt_ * states_k[i + nq];
        res[i + nq] = -states_kp1[i + nq] + states_k[i + nq] + inputs_k[i];
    }
    return 0;
}
Index ocp::eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
{
    const int thread_id = omp_get_thread_num();
    const pinocchio::Model & _model = _pool->getModels()[(size_t)thread_id];
    pinocchio::Data & _data = _pool->getDatas()[(size_t)thread_id];
    pinocchio::Motion & _s_vel = _vel_pool[(size_t)thread_id];

    if (k == K_ - 1) { return 0; }
    else
    {
        Map<const VectorXd> _q  (states_k, nq);
        Map<const VectorXd> _v  (states_k + nq, nq);
        Map<const VectorXd> _a  (inputs_k, nq);

        pinocchio::rnea(_model, _data, _q, _v, _a);
        _s_vel = getFrameVelocity(_model, _data, _ee_frame_id, pinocchio::WORLD);

        for (int i = 0; i < nq; ++i)
        {
            res[i] = _data.tau[i] - inputs_k[nq + i];
        }
        for (int i = 0; i < vel_size; ++i)
        {
            res[nq + i] = _s_vel.toVector()[i]; // - _vel_ref[i]; // vel_ref uncreated yet
        }
    }

    if (k == 0)
    {
        for (int i = 0; i < nq; ++i)
        {
            res[vel_size + nq + i] = states_k[i]; // - ... q0
            res[vel_size + nx + i] = states_k[i + nq]; // - ... v0
        }
    }
    return 0;
};
Index ocp::eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
              Scalar *res, const Index k)
{
    const Scalar s = objective_scale[0];

    if (k == K_ - 1)
    {
        for (int i = 0; i < nx; ++i)
        {
            res[i] = 2 * _Qx_N * states_k[i] * s;
        }
    }
    else
    {
        for (int i = 0; i < nx; ++i)
        {
            res[i] = 2 * _Qu_k * inputs_k[i] * s;
            res[nx + i] = 2 * _Qx_k * states_k[i] * s;
        }
    }
    return 0;
}
Index ocp::eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
             Scalar *res, const Index k)
{
    *res = 0.0;
    const Scalar s = objective_scale[0];

    if (k == K_ - 1)
    {
        for (int i = 0; i < nx; ++i)
        {
            *res+= _Qx_N * states_k[i] * states_k[i] * s;
        }
    }
    else
    {
        for (int i = 0; i < nx; ++i)
        {
            *res+= _Qu_k * inputs_k[i] * inputs_k[i] + _Qx_k * states_k[i] * states_k[i];
        }
        *res*= s;

    }
    return 0;
}
Index ocp::eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
{
    return 0;
};
Index ocp::eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
{
    return 0;
};
Index ocp::get_bounds(Scalar *lower, Scalar *upper, const Index k) const
{
    return 0;
}

Index ocp::get_initial_xk(Scalar *xk, const Index k) const
{
    // xk[0] = 0.;
    // xk[1] = 0.;
    // xk[2] = 0.;
    // xk[3] = 0.;
    return 0;
};
Index ocp::get_initial_uk(Scalar *uk, const Index k) const
{
    // if (k == K_ - 1)
    //     return 0;
    // uk[0] = 0.;
    // uk[1] = 0.;
    return 0;
};
}
