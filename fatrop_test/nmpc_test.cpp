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

using namespace fatrop;

/**
 * This is an example on how to use a specialization for the OcpAbstractTpl class, rather than using
 * the dynamic polymorphism by using the OcpAbstractDynamic tag. Using static polymorphism reduces
 * the overhead and enables compiler optimizations for the NlpOcpTpl class.
 *
 */

// example problem 2D point mass with a small nonlinearity in the dynamics
// states: [x, y, vx, vy]
// inputs: [fx, fy]
// dynamics: [xk+1 = xk + dt*vxk, yk+1 = yk + dt*vyk, vxk+1 = vxk + dt*fx/m +0.5*dt*fy**2/m, vyk+1
// = vyk + dt*fy/m] cost: fx^2 + fy^2
// constraints:
//  at k = 0: x = 0, y = 0, vx = 0, vy = 0
//  at k = K: x = 1, y = 1, vx = 0, vy = 0

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

        void initPinData(const std::string & modelpath, const std::vector<std::string> & list_joints_to_use) {
            _model = dibiman::getArmModel(modelpath, list_joints_to_use);
            _data = pinocchio::Data(_model);
            _pool = std::make_unique<pinocchio::ModelPool>(_model, n_threads);
            nq = _model.nv;
            nx = nq * 2;
            nu = nq * 2;
        };
        void allocateEigenMatrices(const int & nv) {
            nq = nv;
            nx = nv * 2; // q + v : xk
            nu = nv * 2; // a + tau : uk
            // JacobianDerivative.resize(nx, nu);
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
            if (k == 0)
            {
                blasfeo_diare_wrap(nu, 1.f, res, 0, 0);
            }
            else if (k == K_ - 1)
            {
                blasfeo_diare_wrap(nx, 20.f, res, 0, 0);
            }
            else
            {
                blasfeo_diare_wrap(nu, 1.0f, res, 0, 0);
                blasfeo_diare_wrap(nx, 10.0f, res, nu, nu);
            }
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
                Eigen::Map<const Eigen::VectorXd> _q  (states_k,      nq);
                Eigen::Map<const Eigen::VectorXd> _v  (states_k + nq, nq);
                Eigen::Map<const Eigen::VectorXd> _a  (inputs_k,      nq);

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
                    *res+= inputs_k[i] * inputs_k[i];
                }
            }
            else if (k == K_ - 1)
            {
                for (int i = 0; i < nx; ++i)
                {
                    *res+= 10 * states_k[i] * states_k[i];
                }
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                {
                    *res+= inputs_k[i] * inputs_k[i] + 5 * states_k[i] * states_k[i];
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
