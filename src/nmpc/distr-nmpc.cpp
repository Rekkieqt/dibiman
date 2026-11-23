#include <fatrop/fatrop.hpp>
#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "dibiman/nmpc/distr-nmpc.hpp"


using namespace fatrop;

using namespace dibiman;

typedef OcpAbstractTpl<DecentralizedManipulatorStatic> DecentralizedOcp;

DecentralizedNMPC::DecentralizedNMPC(
        const pinocchio::Model & model,
        const std::string & ee_frame_str,
        const int & K,
        const int & n_threads
        )
{
    ocp = std::make_shared<DecentralizedOcp>(model, ee_frame_str, n_threads, K);
    IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcpTpl<DecentralizedManipulatorStatic>>(ocp));
    OptionRegistry options;
    ip_alg = builder.with_options_registry(&options).build();
    ip_data = builder.get_ipdata();
    options.set_option("print_level", 0);

    // trajectory class
    traj = std::make_shared<TrajectoryGenerator>();
    ocp->_traj = traj;
}

void
DecentralizedNMPC::set_dt(const double & dt)
{
    ocp->_dt = dt;
}

void
DecentralizedNMPC::setCosts(const OcpCosts & q)
{
    ocp->_q = q;
}

void
DecentralizedNMPC::setRef(const Eigen::VectorXd & qr)
{
    ocp->xref.head(ocp->nq) = qr;
    ocp->uref.tail(ocp->nv) = pinocchio::rnea(
            ocp->_model, 
            ocp->_data, 
            qr, 
            Eigen::VectorXd::Zero(ocp->nv), 
            Eigen::VectorXd::Zero(ocp->nv));
}
void
DecentralizedNMPC::setWarm()
{
    ocp->_warm = true;
    const auto & info = ip_data->info();
    const VecRealView & sol = ip_data->current_iterate().primal_x();
    // ocp->_warm_x[0] = ocp->x0;
    // pinocchio::rnea(ocp->_model, ocp->_data, 
    //         ocp->x0.head(ocp->nq), // q
    //         ocp->x0.tail(ocp->nv), // v
    //         ocp->x0.tail(ocp->nv)); // reuse v
    // ocp->_warm_u[0] = _data.tau;
    for (int i = 0; i < ocp->_K - 1; ++i)
    {
        const Scalar *x = sol.data() + info.offsets_primal_x[i + 1];
        Eigen::Map<const Eigen::VectorXd> xk_view(x, ocp->nx);
        ocp->_warm_x[i] = xk_view;

        if (i < ocp->_K - 2)
        {
            const Scalar *u = sol.data() + info.offsets_primal_u[i + 1];
            Eigen::Map<const Eigen::VectorXd> uk_view(u, ocp->nu);
            ocp->_warm_u[i] = uk_view;
        }
    }
    const Scalar *x = sol.data() + info.offsets_primal_x[ocp->_K - 1];
    Eigen::Map<const Eigen::VectorXd> xk_view(x, ocp->nx);
    ocp->_warm_x[ocp->_K - 1] = xk_view;
    
    const Scalar *u = sol.data() + info.offsets_primal_u[ocp->_K - 2];
    Eigen::Map<const Eigen::VectorXd> uk_view(u, ocp->nu);
    ocp->_warm_u[ocp->_K - 2] = uk_view;

}

Eigen::VectorXd
DecentralizedNMPC::solve(const Eigen::VectorXd & x0)
{
    ocp->x0 = x0;
    ip_alg->optimize();
    const auto &info = ip_data->info();
    const VecRealView & x_opt = ip_data->current_iterate().primal_x();
    const Scalar *u = x_opt.data() + info.offsets_primal_u[0] + ocp->nv;
    setWarm();

    Eigen::VectorXd tau(ocp->nv); 
    std::memcpy(tau.data(), u, (size_t)ocp->nv * sizeof(Scalar));
    return tau; //Eigen::VectorXd::Zero(ocp->nq);
}

