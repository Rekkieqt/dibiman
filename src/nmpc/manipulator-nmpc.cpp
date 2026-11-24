#include <casadi/casadi.hpp>

#include "dibiman/utils/utils.hpp"
#include "dibiman/utils/robot.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "dibiman/nmpc/manipulator-nmpc.hpp"

using namespace dibiman;

ManipulatorNMPC::ManipulatorNMPC(
        const pinocchio::Model & model,
        const std::string & ee_frame,
        const int & K)
{
    initData(model, ee_frame, K); 
}

void
ManipulatorNMPC::set_dt(const double & dt)
{
    _dt = dt;
}

void
ManipulatorNMPC::setCosts(const OcpCosts & q)
{
    _q = q;
}

void
ManipulatorNMPC::setRef(const Eigen::VectorXd & qr)
{
    xref.head(nq) = qr;
    uref.tail(nv) = pinocchio::rnea(
            _model, 
            _data, 
            qr, 
            Eigen::VectorXd::Zero(nv), 
            Eigen::VectorXd::Zero(nv));
}

void
ManipulatorNMPC::create(void) {  
    using namespace casadi;  

    Function rnea_h = rneaSymbolic(_model);
    Function inv_dyn_f = linearDynamics(nv, _dt);

    SX x0_c = SX::sym("x0", nx), xref_c = SX::sym("xref", nx), uref_c = SX::sym("uref", nv);

    std::vector<SX> X, U, A;
    for (int k = 0; k < _K; ++k) {
        X.push_back(SX::sym("X"+std::to_string(k), nx));
        U.push_back(SX::sym("U"+std::to_string(k), nv));
        A.push_back(SX::sym("A"+std::to_string(k), nv));
    }  

    X.push_back(SX::sym("X"+std::to_string(_K), nx));

    DM Qx = _q.x_k * DM::eye(nx);
    DM Ru = _q.u_k * DM::eye(nv); 
    DM Qa = _q.u_k * DM::eye(nv);  
    DM QxN = _q.x_K * DM::eye(nx);  
  
    SX obj = 0;
    std::vector<SX> g; // Equality constraints
    g.push_back(X[0] - x0_c);

    // Dynamics + Jacobian binding loop  
    for (int k = 0; k < _K; ++k) 
    {  
        SX x_next = inv_dyn_f(std::vector<SX>{X[k], A[k]}).at(0);  
        g.push_back(X[k+1] - x_next);
      
        SX u_k = rnea_h(std::vector<SX>{X[k], A[k]}).at(0);  
        g.push_back(U[k] - u_k); 
    }    

    for (int k = 0; k < _K; ++k) 
    {
        obj += SX::mtimes({(X[k+1] - xref_c).T(), Qx, X[k+1] - xref_c});
        obj += SX::mtimes({(U[k]   - uref_c).T(), Ru, U[k]   - uref_c});
        obj += SX::mtimes({A[k].T(), Qa, A[k]});
    }

    obj += SX::mtimes({(X[_K] - xref_c).T(), QxN, X[_K] - xref_c});
  
    SX X_stack = vertcat(vertcat(X), vertcat(U), vertcat(A));

    SX P = vertcat(x0_c, xref_c, uref_c);
    SX G = vertcat(g);  
  
    SXDict nlp = {{"x", X_stack}, {"f", obj}, {"g", G}, {"p", P}};  
  
    Dict solver_options;  
    solver_options["print_time"] = false;  
    solver_options["ipopt.print_level"] = 0;  
  
    solver = nlpsol("solver", "ipopt", nlp, solver_options);  

    /* __________________________________ Solver Init Testing __________________________ */

    auto solverInfo = [](const casadi::Function& solver) {  
        if (solver.is_null()) 
        {  
            std::cout << "Solver NOT initialized (null Function)\n";  
        } 
        else 
        {  
            std::cout << "Solver: " << solver.name()  
                       << " (n_in=" << solver.n_in() << ", n_out=" << solver.n_out() << ")\n";  
            for (casadi_int i = 0; i < solver.n_in(); ++i) 
            {  
                std::cout << "  in[" << i << "] " << solver.name_in(i) << ": "  
                           << solver.size1_in(i) << "x" << solver.size2_in(i) << "\n";  
            }  
            for (casadi_int i = 0; i < solver.n_out(); ++i) 
            {  
                std::cout << "  out[" << i << "] " << solver.name_out(i) << ": "  
                           << solver.size1_out(i) << "x" << solver.size2_out(i) << "\n";  
            }  
        }
    };
    solverInfo(solver);
};

Eigen::VectorXd
ManipulatorNMPC::solve(const Eigen::VectorXd & xk)
{  
    using namespace casadi;

    std::vector<DM> p_val_list;
    const DM xk_dm = eigenToDM(xk);
    const DM xref_dm = eigenToDM(xref);
    const DM uref_dm = eigenToDM(uref.tail(nv));

    p_val_list.push_back(xk_dm);
    p_val_list.push_back(xref_dm);
    p_val_list.push_back(uref_dm);

    casadi_int ng = solver.size1_in(solver.index_in("lbg"));
    DM lbg = DM::zeros(ng);
    DM ubg = DM::zeros(ng);

    DM p_val = vertcat(p_val_list);

    DMDict arg = {{"p", p_val}, {"lbg", lbg}, {"ubg", ubg}};
    DMDict res = solver(arg);

    DM x_opt = res.at("x");

    /*
    0, (H+1)*nx = Xr
    (H+1)*nx, (H+1)*nx + H*nv = Ur
    (H+1)*nx + H*nv, (H+1)*nx + H*nv + H*nv = Ar 
    (H+1)*nx + H*nv + H*nv, (H+1)*nx + H*nv + H*nv + (H+1)*nx = Xl
    (H+1)*nx + H*nv + H*nv + (H+1)*nx, (H+1)*nx + H*nv + H*nv + (H+1)*nx + H*nv = Ul
    (H+1)*nx + H*nv + H*nv + (H+1)*nx + H*nv, (H+1)*nx + H*nv + H*nv + (H+1)*nx + H*nv + H*nv = Al
    */

    casadi_int offset_X = 0;
    casadi_int offset_U = (_K + 1) * nx;
    casadi_int offset_A = offset_U + _K * nv;
    DM uk_dm = x_opt(Slice(offset_U, offset_U + nv));

    Eigen::VectorXd uk(uref.size());
    std::copy(uk_dm.ptr(), uk_dm.ptr() + uk_dm.numel(), uk.data());
    return  uk;
};
