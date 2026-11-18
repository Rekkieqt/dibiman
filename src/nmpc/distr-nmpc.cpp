#include <limits>
#include <filesystem>
#include <memory>
#include <omp.h>
#include <fatrop/fatrop.hpp>
// #include "fatrop/ocp/nlp_ocp.hxx"
#include "nlp_ocp.hxx"
#include "dibiman/utils/robot.hpp"

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
// #include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"

// #include "pinocchio/algorithm/jacobian.hpp"
// #include "pinocchio/algorithm/joint-configuration.hpp"
// #include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/spatial.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

using namespace fatrop;
using namespace Eigen;

class RobotModel
{
    public:
        const Scalar dt_ = 0.05;

        Matrix3d Ri = Matrix3d::Identity();
        Vector3d pi = Vector3d(0, 0, 0);
        pinocchio::SE3 Xi{Ri, pi};

        Matrix3d Rf = pinocchio::rpy::rpyToMatrix(Vector3d(M_PI/2.0, 0, 0));
        Vector3d pf = Vector3d(1, 1, 1);
        pinocchio::SE3 Xf{Rf, pf};

    protected:
        int nx;
        int nu;
        int nq;
        int nv;

        const int vel_size = 6;

        const double a5 = 6;
        const double a4 = -15;
        const double a3 = 10;

        const double Tf = 10;

        pinocchio::FrameIndex _ee_frame_id;

        pinocchio::Model _model;
        pinocchio::Data _data;
        std::unique_ptr<pinocchio::ModelPool> _pool;


        const double _Qx_k = 1;
        const double _Qv_k = 1;

        const double _Qa_k = 1;
        const double _Qu_k = 1;

        const double _Qx_N = 10;
        const double _Qv_N = 10;

        struct _frame_velocity_derivatives {
            pinocchio::Data::Matrix6x dq;
            pinocchio::Data::Matrix6x dv;

            _frame_velocity_derivatives() = default;

            _frame_velocity_derivatives(const int & vel_size, const int & nv) :
                dq(vel_size, nv),
                dv(vel_size, nv)
            {}

            void setZero(void)
            {
                dq.setZero();
                dv.setZero();
            }
        };

        struct _rnea_second_order_matrices {
            Tensor<double, 2> _qq;
            Tensor<double, 2> _qv;
            Tensor<double, 2> _vv;
            Tensor<double, 2> _aq;

            _rnea_second_order_matrices() = default;

            _rnea_second_order_matrices(const int nq) :
                _qq(nq, nq),
                _qv(nq, nq),
                _vv(nq, nq),
                _aq(nq, nq)
            {}

            void setZero(void)
            {
                _qq.setZero();
                _qv.setZero();
                _vv.setZero();
                _aq.setZero();
            }
        };

        std::vector<_rnea_second_order_matrices> _mat_pool;
        std::vector<_frame_velocity_derivatives> _vel_partial_pool;
        std::vector<pinocchio::Motion> _vel_pool;

        void initPinData(const std::string & modelpath, const std::vector<std::string> & list_joints_to_use, const int & n_threads) {
            _model = dibiman::getArmModel(modelpath, list_joints_to_use);
            _data = pinocchio::Data(_model);
            _pool = std::make_unique<pinocchio::ModelPool>(_model, n_threads);
            nq = _model.nq;
            nv = _model.nv;
            nx = nq * 2;
            nu = nq * 2;
            _mat_pool.reserve(n_threads);
            _ee_frame_id = _model.getFrameId("r_hand_dh_frame");
            for (int i = 0; i < n_threads; ++i)
            {
                _mat_pool.emplace_back(nq);
                _vel_partial_pool.emplace_back(vel_size, nv);
                _vel_pool.emplace_back();
            }
        }

        inline double get_s(const double & t, const double & Tf)
        {

            double x = t / Tf;
            double x3 = x * x * x;
            return a3 * x3 + a4 * x3 * x + a5 * x3 * x * x;
        }

        inline double get_ds(const double & t, const double & Tf)
        {
            double x = t / Tf;
            double x2 = x * x;
            double res = 3 * a3 * x2 + 4 * a4 * x2 * x + 5 * a5 * x2 * x2;
            return res * (1.0)/Tf;
        }

        inline pinocchio::Motion twist_ref(const fatrop::Index k)
        {
            double s = get_s(k * dt_, Tf);
            double ds = get_ds(k * dt_, Tf);
            pinocchio::Motion nu = pinocchio::log6(Xi.actInv(Xf));
            pinocchio::SE3 Xs = Xi * pinocchio::exp6(nu * s);
            return Xs.act(nu) * ds;
        }

};

struct RobotManipulator
{
};

typedef fatrop::OcpAbstractTpl<RobotManipulator> ManipulatorOcp;

template <> 
class fatrop::OcpAbstractTpl<RobotManipulator> : public RobotModel
{
    /* nx = [q, v]
     * nu = [a, tau]
     */
    public:
        OcpAbstractTpl(const std::string & modelpath, const std::vector<std::string> & joints, const int & n_threads)
        {
            initPinData(modelpath, joints, n_threads);
        }

        Index get_nx(const Index k) const { return nx; }

        Index get_nu(const Index k) const
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

        Index get_ng(const Index k) const
        {
            if (k == 0)
            {
                return vel_size + nx + nq;
            }
            else if (k == K_ - 1)
            {
                return 0;
            }
            else
            {
                return nq + vel_size;
            }
        };

        Index get_ng_ineq(const Index k) const { return 0; };
        Index get_horizon_length() const { return K_; };
        Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
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
        Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
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
            /*
            const int thread_id = omp_get_thread_num();
            const pinocchio::Model & _model = _pool->getModels()[(size_t)thread_id];
            pinocchio::Data & _data = _pool->getDatas()[(size_t)thread_id];
            _rnea_second_order_matrices & _m = _mat_pool[(size_t)thread_id];
            _m.setZero();

            _m._qq.diagonal().array() += 2 * _Qx_k;
            _m._vv.diagonal().array() += 2 * _Qx_k;

            // Put RNEA and Jacobian Derivatives here
            Map<const VectorXd> _q  (states_k, nq);
            Map<const VectorXd> _v  (states_k + nq, nq);
            Map<const VectorXd> _a  (inputs_k, nq);
            TensorMap<const Tensor<double, 1>> _lam_eq  (lam_eq_k, nq);

            ComputeRNEASecondOrderDerivatives(_model, _data, _q, _v, _a);

            // Tensor contraction
            const Eigen::array<Eigen::IndexPair<int>, 1> dims = { Eigen::IndexPair<int>(0, 0) };
            _m._qq = _lam_eq.contract(_data.d2tau_dqdq, dims);
            _m._vv = _lam_eq.contract(_data.d2tau_dvdv, dims);
            _m._qv = _lam_eq.contract(_data.d2tau_dqdv, dims);
            _m._aq = _lam_eq.contract(_data.d2tau_dadq, dims);

            Map<MatrixXd> _qq_mat(_m._qq.data(), nq, nq);
            Map<MatrixXd> _vv_mat(_m._vv.data(), nq, nq);
            Map<MatrixXd> _qv_mat(_m._qv.data(), nq, nq);
            Map<MatrixXd> _aq_mat(_m._aq.data(), nq, nq);

            _qq_mat.diagonal().array() += s * 2 * _Qx_k;
            _vv_mat.diagonal().array() += s * 2 * _Qx_k;

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
        Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
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

            // computeForwardKinematicsDerivatives(_model, _data, _q, _v);
            // _data.J;
            // _data.dJ;
            if (k == 0)
            {
                blasfeo_diare_wrap(nx, 1.f, res, nu, nq + vel_size);
            }
            return 0;
        }
        Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
                     Scalar *res, const Index k)
        {
            // set zero fo k == 0 also sets the initial state to zero
            // blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            for (int i = 0; i < nq; ++i)
            {
                res[i] = -states_kp1[i] + states_k[i] + dt_ * states_k[i + nq];
                res[i + nq] = -states_kp1[i + nq] + states_k[i + nq] + inputs_k[i];
            }
            return 0;
        }
        Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
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
                // updateFramePlacements(_model, _data);
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
                    // initial state is 0
                    res[vel_size + nq + i] = states_k[i]; // - ... q0
                    res[vel_size + nx + i] = states_k[i + nq]; // - ... v0
                }
            }
            return 0;
        };
        Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
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
        Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
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
        Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
        {
            return 0;
        };
        Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
        {
            return 0;
        };
        Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
        {
            return 0;
        }

        Index get_initial_xk(Scalar *xk, const Index k) const
        {
            // xk[0] = 0.;
            // xk[1] = 0.;
            // xk[2] = 0.;
            // xk[3] = 0.;
            return 0;
        };
        Index get_initial_uk(Scalar *uk, const Index k) const
        {
            // if (k == K_ - 1)
            //     return 0;
            // uk[0] = 0.;
            // uk[1] = 0.;
            return 0;
        };

    private:
        const Index K_ = 50;
};

int main()
{
    const std::string modelpath = static_cast<std::string>(std::filesystem::current_path()) + "/../../conf/model.urdf";
    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};
    constexpr int n_threads = 4;
    omp_set_num_threads(n_threads);

    auto problem = std::make_shared<ManipulatorOcp>(modelpath, joint_list, n_threads);

    IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcpTpl<RobotManipulator>>(problem));

    std::shared_ptr<IpAlgorithm<OcpType>> ipalg = builder.build();
    IpSolverReturnFlag ret = ipalg->optimize();
    auto data = builder.get_ipdata();
    std::cout << "Return flag: " << int(ret) << std::endl;
    std::cout << "Return flag == success: " << (ret == IpSolverReturnFlag::Success) << std::endl;
    std::cout << data->timing_statistics() << std::endl;
    return 0;
}

