#include <limits>
#include <filesystem>
#include <memory>
#include <fatrop/fatrop.hpp>
#include "fatrop/ocp/nlp_ocp.hxx"
#include "dibiman/utils/robot.hpp"

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
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
        int nx;
        int nu;

        // Pinocchio preallocated data
        Eigen::VectorXd _q;
        Eigen::VectorXd _v;
        Eigen::VectorXd _a;
        Eigen::VectorXd _tau;

        // RNEA 1st derivative
        Eigen::MatrixXd djoint_torque_dq;
        Eigen::MatrixXd djoint_torque_dv;
        Eigen::MatrixXd djoint_torque_da;
        //
        // Jacobian 1st derivative 
        // Eigen::MatrixXd JacobianDerivative;
        //
        void initPinData(const std::string & modelpath, const std::vector<std::string> & list_joints_to_use) {
            _model = dibiman::getArmModel(modelpath, list_joints_to_use);
            _data = pinocchio::Data(_model);
        };
        void allocateEigenMatrices(const int & nv) {
            nx = nv * 2; // q + v : state
            nu = nv * 2; // a + tau : control

            _q.resize(nq);
            _v.resize(nq);
            _a.resize(nq);
            _tau.resize(nq);

            djoint_torque_dq.resize(nq);
            djoint_torque_dv.resize(nq);
            djoint_torque_da.resize(nq);

            // rneaDerivative.resize(nx, nu);
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
            allocateEigenMatrices(_model.nv);
        }

        Index get_nx(const Index k) const { return nx; }

        Index get_nu(const Index k) const
        {
            if (k == 0)
            {
                return nu;
            }
            else if (k == K_ - 1)
            {
                return 0;
            }
            else
            {
                return nu;
            }
        }

        /**
         * @brief Get the number of inequality constraints for a given time step.
         * @param k Time step
         * @return Number of inequality constraints
         */

        Index get_ng(const Index k) const
        {
            if (k == 0)
            {
                return _model.nv;
            }
            else if (k == K_ - 1)
            {
                return _model.nv;
            }
            else
            {
                return 0;
            }
        };

        Index get_ng_ineq(const Index k) const { return 0; };
        Index get_horizon_length() const { return K_; };
        Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
                        MAT *res, const Index k)
        {
            // set zero
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            // Matrix B
            // [  0,    0        ]
            // [  0,    0        ]
            // [ dt/m,  dt/m*fy  ]
            // [  0,    dt/m     ]

            // Matrix A
            // [ 1,  0,  dt,  0  ]
            // [ 0,  1,   0,  dt ]
            // [ 0,  0,   1,  0  ]
            // [ 0,  0,   0,  1  ]

            blasfeo_matel_wrap(res, 0, 2) = dt_ / m_;
            blasfeo_matel_wrap(res, 1, 3) = dt_ / m_;
            blasfeo_matel_wrap(res, 1, 2) = dt_ / m_ * (inputs_k[1] - 1.);

            blasfeo_diare_wrap(4, 1.0, res, 2, 0);
            blasfeo_matel_wrap(res, 4, 0) = dt_;
            blasfeo_matel_wrap(res, 5, 1) = dt_;
            return 0;
        }
        Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                          const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k,
                          MAT *res, const Index k)
        {
            // set zero
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            // Matrix R
            // [ 2,  0 ]
            // [ 0,  2 ]
            if (k < K_ - 1)
            {
                blasfeo_diare_wrap(2, objective_scale[0] * 2.0, res, 0, 0);
                // add the contribution from the nonlinearity in the dynamics
                Scalar lam = lam_dyn_k[2];
                blasfeo_matel_wrap(res, 1, 1) += dt_ * lam / m_;
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
            // Put RNEA and Jacobian Derivatives here
            computeRNEADerivatives(_model, _data, q, v, a, djoint_torque_dq, djoint_torque_dv, djoint_torque_da);

            /* RNEA constraint
             * h(q, v, a, tau) = rnea(q, v, a) - tau = 0
             * dh/da, dh/dtau, dh/dq, dh/dv = dh/d(uk, xk)
             * dh/dtau = I_nq 
             *
             * Jacobian constraint
             * h(q, v, a, tau) = J(q) * v - (numerical v_des)
             * dh/da, dh/dtau, dh/dq, dh/da = dh/d(uk, xk)
             * dh/da = 0
             * dh/dtau = 0
             * dh/duk = 0
             */

            // set zero
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            if (k == 0)
                blasfeo_diare_wrap(4, 1.0, res, 2, 0);
            if (k == K_ - 1)
                blasfeo_diare_wrap(4, 1.0, res, 0, 0);
            return 0;
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
            res[0] = -states_kp1[0] + states_k[0] + dt_ * states_k[2];
            res[1] = -states_kp1[1] + states_k[1] + dt_ * states_k[3];
            res[2] = -states_kp1[2] + states_k[2] + dt_ * inputs_k[0] / m_ +
                     dt_ * 0.5 * (inputs_k[1] - 1.) * (inputs_k[1] - 1.) / m_;
            res[3] = -states_kp1[3] + states_k[3] + dt_ * inputs_k[1] / m_;
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
            _tau = pinocchio::rnea(_model, _data, _q, _v, _a);
            if (k == 0)
            {
                res[0] = states_k[0];
                res[1] = states_k[1];
                res[2] = states_k[2];
                res[3] = states_k[3];
            }
            else if (k == K_ - 1)
            {
                res[0] = states_k[0] - 1.0;
                res[1] = states_k[1] - 2.0;
                res[2] = states_k[2] - 3.0;
                res[3] = states_k[3] - 4.0;
            }
            return 0;
        };
        Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                      Scalar *res, const Index k)
        {
            if (k == K_ - 1)
            {
                res[0] = 0.;
                res[1] = 0.;
                res[2] = 0.;
                res[3] = 0.;
            }
            else
            {
                res[0] = 2 * objective_scale[0] * inputs_k[0];
                res[1] = 2 * objective_scale[0] * inputs_k[1];
                res[2] = 0.;
                res[3] = 0.;
                res[4] = 0.;
                res[5] = 0.;
                res[6] = 0.;
            }
            return 0;
        }
        Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                     Scalar *res, const Index k)
        {
            if (k == K_ - 1)
            {
                *res = 0.;
            }
            else
            {
                *res = objective_scale[0] * (inputs_k[0] * inputs_k[0] + inputs_k[1] * inputs_k[1]);
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
            xk[0] = 0.;
            xk[1] = 0.;
            xk[2] = 0.;
            xk[3] = 0.;
            return 0;
        };
        Index get_initial_uk(Scalar *uk, const Index k) const
        {
            if (k == K_ - 1)
                return 0;
            uk[0] = 0.;
            uk[1] = 0.;
            return 0;
        };

    private:
        const Index K_ = 100;
        const Scalar m_ = 1.0;
        const Scalar dt_ = 0.05;
};

int main()
{
    const std::string modelpath = static_cast<std::string>(std::filesystem::current_path()) + "/../../conf/model.urdf";
    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};
    auto problem = std::make_shared<ManipulatorOcp>(modelpath, joint_list);

    IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcpTpl<RobotManipulator>>(problem));

    std::shared_ptr<IpAlgorithm<OcpType>> ipalg = builder.build();
    IpSolverReturnFlag ret = ipalg->optimize();
    std::cout << "Return flag: " << int(ret) << std::endl;
    std::cout << "Return flag == success: " << (ret == IpSolverReturnFlag::Success) << std::endl;
    return 0;
}
