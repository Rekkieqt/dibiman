#include <fatrop/fatrop.hpp>
#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/parallel/rnea.hpp"
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

Eigen::VectorXd
DecentralizedNMPC::solve(const Eigen::VectorXd & x0)
{
    ocp->x0 = x0;
    ip_alg->optimize();
    const auto &info = ip_data->info();
    const VecRealView & x_opt = ip_data->current_iterate().primal_x();
    const Scalar *u = x_opt.data() + info.offsets_primal_u[0];

    Eigen::VectorXd tau(ocp->nv); 
    std::memcpy(tau.data(), u, (size_t)ocp->nv * sizeof(Scalar));
    return tau; //Eigen::VectorXd::Zero(ocp->nq);
}

