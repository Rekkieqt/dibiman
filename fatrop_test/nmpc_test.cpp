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
#include "pinocchio/algorithm/parallel/rnea.hpp"
// #include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"

using namespace fatrop;
using namespace Eigen;


/**
 * This is an example on how to use a specialization for the OcpAbstractTpl class, rather than using
 * the dynamic polymorphism by using the OcpAbstractDynamic tag. Using static polymorphism reduces
 * the overhead and enables compiler optimizations for the NlpOcpTpl class.
 *
 */

class RobotModel
{
    protected:
        pinocchio::Model _model;
        pinocchio::Data _data;
        std::unique_ptr<pinocchio::ModelPool> _pool;

        int nx;
        int nu;
        int nq;
        const size_t n_threads = 8;

        struct _rnea_second_order_matrices {
            MatrixXd _qq;
            MatrixXd _qv;
            MatrixXd _vv;
            MatrixXd _aq;

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

        // cost matrices
        const double _Qq_k = 10;
        const double _Qv_k = 5;

        const double _Qa_k = 1;
        const double _Qtau_k = 1;

        const double _Qq_N = 50;
        const double _Qv_N = 25;

        void initPinData(const std::string & modelpath, const std::vector<std::string> & list_joints_to_use) {
            _model = dibiman::getArmModel(modelpath, list_joints_to_use);
            _data = pinocchio::Data(_model);
            _pool = std::make_unique<pinocchio::ModelPool>(_model, n_threads);
            nq = _model.nv;
            nx = nq * 2;
            nu = nq * 2;
            _mat_pool.reserve(n_threads);
            for (int i = 0; i < n_threads; ++i)
            {
                _rnea_second_order_matrices temp(nq);
                _mat_pool.push_back(temp);
                //_mat_pool.emplace(i, nq);
            }
        };
        // void allocateEigenMatrices(const int & nv) {}

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
        OcpAbstractTpl(const std::string & modelpath, const std::vector<std::string> & joints)
        {
            initPinData(modelpath, joints);
            /* allocateEigenMatrices(_model.nv); */
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
                return nx + nq;
            }
            else if (k == K_ - 1)
            {
                return 0;
            }
            else
            {
                return nq;
            }
        };

        Index get_ng_ineq(const Index k) const { return 0; };
        Index get_horizon_length() const { return K_; };
        Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
                        MAT *res, const Index k)
        {
            /* from original example
            col:            0      1      2        3       ← next-state index (transposed!)
            (u0) row 0  [   0      0     dt/m      0   ]   ← B^T  
            (u1) row 1  [   0      0     dt/m(u1-1) dt/m]  
            (x0) row 2  [   1      0      0        0   ]  
            (x1) row 3  [   0      1      0        0   ]   ← A^T  
            (x2) row 4  [   dt     0      1        0   ]  
            (x3) row 5  [   0      dt     0        1   ]             
            */
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
            if (k == K_ - 1)
            {
                blasfeo_diare_wrap(nx, 2 * _Qq_N, res, 0, 0);
                return 0;
            }

            const int thread_id = omp_get_thread_num();
            const pinocchio::Model & _model = _pool->getModels()[(size_t)thread_id];
            pinocchio::Data & _data = _pool->getDatas()[(size_t)thread_id];
            _rnea_second_order_matrices & _m = _mat_pool[(size_t)thread_id];
            _m.setZero();

            /*
            Map<MatrixXd> _qq_mat(_m._qq.data(), nq, nq);
            Map<MatrixXd> _vv_mat(_m._vv.data(), nq, nq);
            Map<MatrixXd> _qv_mat(_m._qv.data(), nq, nq);
            Map<MatrixXd> _aq_mat(_m._aq.data(), nq, nq);
            */

            if (k == 0)
            {
                blasfeo_diare_wrap(nu, 2 * _Qtau_k, res, 0, 0);
            }
            else
            {
                blasfeo_diare_wrap(nu, 2 * _Qtau_k, res, 0, 0);
                blasfeo_diare_wrap(nx, 2 * _Qq_k, res, nu, nu);
                // _m._qq.diagonal().array() += 2 * _Qq_k;
                // _m._vv.diagonal().array() += 2 * _Qq_k;
            }

            // Put RNEA and Jacobian Derivatives here
            Map<const VectorXd> _q  (states_k, nq);
            Map<const VectorXd> _v  (states_k + nq, nq);
            Map<const VectorXd> _a  (inputs_k, nq);
            Map<const VectorXd> _lam_eq  (lam_eq_k, nq);
            // TensorMap<const Tensor<double, 1>> _lam_eq  (lam_eq_k, nq);

            ComputeRNEASecondOrderDerivatives(_model, _data, _q, _v, _a);
            for (int i = 0; i < nq; ++i)
            {
                for (int j = 0; j < nq; ++j)
                {
                    for (int kk = 0; kk < nq; ++kk)
                    {
                        _m._qq(j, kk) += lam_eq_k[i] * _data.d2tau_dqdq(i, j, kk);
                        _m._qv(j, kk) += lam_eq_k[i] * _data.d2tau_dqdv(i, j, kk);
                        _m._vv(j, kk) += lam_eq_k[i] * _data.d2tau_dvdv(i, j, kk);
                        _m._aq(j, kk) += lam_eq_k[i] * _data.d2tau_dadq(i, j, kk);
                    }
                }
            }

            /* Tensor contraction
            const Eigen::array<Eigen::IndexPair<int>, 1> dims = { Eigen::IndexPair<int>(0, 0) };
            _m._qq = _lam_eq.contract(_data.d2tau_dqdq, dims);
            _m._vv = _lam_eq.contract(_data.d2tau_dvdv, dims);
            _m._qv = _lam_eq.contract(_data.d2tau_dqdv, dims);
            _m._aq = _lam_eq.contract(_data.d2tau_dadq, dims);
            */

            /* Deep Wiki variation on pinocchio unittest
            Map<const MatrixXd> T_qq(_data.d2tau_dqdq.data(), nq, nq * nq);  
            Map<const MatrixXd> T_vv(_data.d2tau_dvdv.data(), nq, nq * nq);  
            Map<const MatrixXd> T_qv(_data.d2tau_dqdv.data(), nq, nq * nq);  
            Map<const MatrixXd> T_aq(_data.d2tau_dadq.data(), nq, nq * nq);  

            Map<RowVectorXd>(_m._qq.data(), nq*nq) = _lam_eq.transpose() * T_qq;
            Map<RowVectorXd>(_m._vv.data(), nq*nq) = _lam_eq.transpose() * T_vv;
            Map<RowVectorXd>(_m._qv.data(), nq*nq) = _lam_eq.transpose() * T_qv;
            Map<RowVectorXd>(_m._aq.data(), nq*nq) = _lam_eq.transpose() * T_aq;
            */

            blasfeo_pack_dmat(nq, nq,
                    _m._qq.data(), _m._qq.outerStride(), 
                    res, nx, nx);
            blasfeo_pack_dmat(nq, nq,
                    _m._vv.data(), _m._vv.outerStride(), 
                    res, nx + nq, nx + nq);
            blasfeo_pack_dmat(nq, nq,
                    _m._qv.data(), _m._qv.outerStride(), 
                    res, nx, nx + nq);
            blasfeo_pack_dmat(nq, nq,
                    _m._aq.data(), _m._aq.outerStride(), 
                    res, 0, nx);

            blasfeo_pack_tran_dmat(nq, nq,
                    _m._qv.data(), _m._qv.outerStride(), 
                    res, nx + nq, nx);
            blasfeo_pack_tran_dmat(nq, nq,
                    _m._aq.data(), _m._aq.outerStride(), 
                    res, nx, 0);


            return 0;
        };
        /**
         * @brief Evaluate the stagewise equality constraints Jacobian.
         *
         * This method evaluates the Jacobian of the equality constraints
         * with respect to the concatenation of (u_k, x_k).
         * The matrix is a BLASFEO matrix type.
         *
         * Note: The last row of the matrix is used internally and should not
         * be evaluated by the user.
         *
         * @param inputs_k Pointer to array inputs of time step k
         * @param states_k Pointer to array states of time step k
         * @param res Pointer to (nu+nx+1 x ng)-matrix
         * @param k Time step
         * @return Status code
         */
        Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k)
        {
            if (k == K_ - 1) return 0;

            const int thread_id = omp_get_thread_num();
            const pinocchio::Model & _model = _pool->getModels()[(size_t)thread_id];
            pinocchio::Data & _data = _pool->getDatas()[(size_t)thread_id];

            // Put RNEA and Jacobian Derivatives here
            Eigen::Map<const Eigen::VectorXd> _q  (states_k,      nq);
            Eigen::Map<const Eigen::VectorXd> _v  (states_k + nq, nq);
            Eigen::Map<const Eigen::VectorXd> _a  (inputs_k,      nq);
            computeRNEADerivatives(_model, _data, _q, _v, _a);

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

            if (k == 0)
            {
                // wrong index
                blasfeo_diare_wrap(nx, 1.f, res, nu, nq); // ai check, apparently its all transpose so additional h(uk, xk) are columns
            }
            return 0;
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
        }

        /**
         * This method evaluates b_k = -x_{k+1} + f_k(u_k, x_k, p_k, p).
         *
         * @param states_kp1 Pointer to array states of time step k+1
         * @param inputs_k Pointer to array inputs of time step k
         * @param states_k Pointer to array states of time step k
         * @param res Pointer to array nx_{k+1}-vector
         */

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

        /**
         * @brief Evaluate the equality constraint violation.
         *
         * This method evaluates g_k = g_k(u_k, x_k, p_k, p).
         *
         * @param inputs_k Pointer to array inputs of time step k
         * @param states_k Pointer to array states of time step k
         * @param res Pointer to array ng-vector
         * @param k Time step
         * @return Status code
         */

        Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k)
        {
            const int thread_id = omp_get_thread_num();
            const pinocchio::Model & _model = _pool->getModels()[(size_t)thread_id];
            pinocchio::Data & _data = _pool->getDatas()[(size_t)thread_id];

            if (k == K_ - 1) { return 0; }
            else
            {
                Eigen::Map<const Eigen::VectorXd> _q  (states_k, nq);
                Eigen::Map<const Eigen::VectorXd> _v  (states_k + nq, nq);
                Eigen::Map<const Eigen::VectorXd> _a  (inputs_k, nq);

                pinocchio::rnea(_model, _data, _q, _v, _a);
                for (int i = 0; i < nq; ++i)
                {
                    res[i] = _data.tau[i] - inputs_k[nq + i];
                }
            }

            if (k == 0)
            {
                for (int i = 0; i < nq; ++i)
                {
                    // initial state is 0
                    res[nq + i] = states_k[i]; // - ...
                    res[nx + i] = states_k[i + nq]; // - ...
                }
            }
            return 0;
        };
        Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                      Scalar *res, const Index k)
        {
            /**
            * @brief Evaluate the gradient of the objective function.
            *
            * This method evaluates the gradient of the objective function (not the Lagrangian!)
            * with respect to the concatenation of (u_k, x_k).
            *
            * @param objective_scale Pointer to objective scale
            * @param inputs_k Pointer to array inputs of time step k
            * @param states_k Pointer to array states of time step k
            * @param res Pointer to (nu+nx)-array
            * @param k Time step
            * @return Status code
            */
            // blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            if (k == K_ - 1)
            {
                for (int i = 0; i < nx; ++i)
                {
                    res[i] = 20 * states_k[i];
                }
            }
            else if (k == 0)
            {
                for (int i = 0; i < nu; ++i)
                {
                    res[i] = 2 * inputs_k[i];
                    res[nu + i] = 10 * states_k[nu + i];
                }
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                {
                    res[i] = 2 * inputs_k[i];
                    res[nx + i] = 10 * states_k[i];
                }
            }
            return 0;
        }
        Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                     Scalar *res, const Index k)
        {
            *res = 0.0;
            if (k == 0)
            {
                for (int i = 0; i < nx; ++i)
                {
                    *res+= _Qtau_k * inputs_k[i] * inputs_k[i];
                }
            }
            else if (k == K_ - 1)
            {
                for (int i = 0; i < nx; ++i)
                {
                    *res+= _Qq_N * states_k[i] * states_k[i];
                }
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                {
                    *res+= _Qtau_k * inputs_k[i] * inputs_k[i] + _Qq_k * states_k[i] * states_k[i];
                }

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
        const Scalar dt_ = 0.05;
};

int main()
{
    const std::string modelpath = static_cast<std::string>(std::filesystem::current_path()) + "/../../conf/model.urdf";
    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};
    constexpr int n_threads = 4;
    omp_set_num_threads(n_threads);

    auto problem = std::make_shared<ManipulatorOcp>(modelpath, joint_list);

    IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcpTpl<RobotManipulator>>(problem));

    std::shared_ptr<IpAlgorithm<OcpType>> ipalg = builder.build();
    IpSolverReturnFlag ret = ipalg->optimize();
    auto data = builder.get_ipdata();
    std::cout << "Return flag: " << int(ret) << std::endl;
    std::cout << "Return flag == success: " << (ret == IpSolverReturnFlag::Success) << std::endl;
    std::cout << data->timing_statistics() << std::endl;
    return 0;
}
