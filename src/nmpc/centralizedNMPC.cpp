#include "dibiman/nmpc/baseNMPC.hpp"
#include <casadi/casadi.hpp>

using namespace dibiman;

centralizedNMPC::centralizedNMPC(
        const std::string & model_path,
        const std::vector<std::string>& joints_to_use,
        const std::vector<std::string>& arm_prefixes
        )
{
    for (const auto& prefix : arm_prefixes) 
    {
        addArmToList(model_path, joints_to_use, prefix);
    }
};

void 
centralizedNMPC::rneaSolver(void) {  
    using namespace casadi;  
  
    for (int k = 0; k < H; ++k) {  
        Xr.push_back(optimizer.variable(nx));  
        Ur.push_back(optimizer.variable(nu));  
        Ar.push_back(optimizer.variable(na));  

        Xl.push_back(optimizer.variable(nx));  
        Ul.push_back(optimizer.variable(nu));  
        Al.push_back(optimizer.variable(na));  
    }  

    Xr.push_back(optimizer.variable(nx));  
    Xl.push_back(optimizer.variable(nx));  
  
    // Cost function  
    DM R  = 1 * DM::eye(nu);  
    DM Qx = 7 * DM::eye(nx);  
    DM Qa = DM::eye(na);  
  
    MX obj = 0;  
    for (int i = 0; i < H; ++i) {  
        obj += mtimes({(Xr[i+1] - xref_r).T(), Qx, Xr[i+1] - xref_r});  
        obj += mtimes({(Ur[i]   - uref_r).T(), R,  Ur[i]   - uref_r});  
        obj += mtimes({Ar[i].T(), Qa, Ar[i]});  
  
        obj += mtimes({(Xl[i+1] - xref_l).T(), Qx, Xl[i+1] - xref_l});  
        obj += mtimes({(Ul[i]   - uref_l).T(), R,  Ul[i]   - uref_l});  
        obj += mtimes({Al[i].T(), Qa, Al[i]});  
    }  
  
    optimizer.subject_to(Xr[0] == x0_r);  
    optimizer.subject_to(Xl[0] == x0_l);  
  
    for (int k = 0; k < H; ++k) {  
        /* needs to be redone */
        /*
        optimizer.subject_to(Xr[k+1] == inverseModel(nq)(std::vector<MX>{Xr[k], Ar[k]}).at(0));  
        optimizer.subject_to(Ur[k]   == rnea(models[0])(std::vector<MX>{Xr[k], Ar[k]}).at(0));  
  
        optimizer.subject_to(Xl[k+1] == inverseModel(nq)(std::vector<MX>{Xl[k], Al[k]}).at(0));  
        optimizer.subject_to(Ul[k]   == rnea(models[1])(std::vector<MX>{Xl[k], Al[k]}).at(0));  
  
        MX err = armJacobian(models[0], handIds[0])(std::vector<MX>{Xr[k+1]}).at(0)  
               - armJacobian(models[1], handIds[1])(std::vector<MX>{Xl[k+1]}).at(0);  
        optimizer.subject_to(err == 0);  
        */
    }  
  
    Dict solver_options;
    solver_options["print_time"] = false;
    solver_options["expand"] = true;
    solver_options["ipopt.print_level"] = true;
    solver_options["ipopt.tol"] = 1e3f;
    optimizer.solver("ipopt", solver_options);  
};

void
centralizedNMPC::solve(
    const casadi::DM& x0r, const casadi::DM& x0l,  
    const casadi::DM& xRefr, const casadi::DM& xRefl,  
    const casadi::DM& uRefr, const casadi::DM& uRefl
    ) 
{  
    using namespace casadi;
  
    optimizer.set_value(x0_r, x0r);
    optimizer.set_value(xref_r, xRefr);  
    optimizer.set_value(uref_r, uRefr);  
  
    optimizer.set_value(x0_l, x0l);
    optimizer.set_value(xref_l, xRefl);
    optimizer.set_value(uref_l, uRefl);
  
    OptiSol solution = optimizer.solve();  
  
    DM u_r_star = solution.value(Ur[0]);  
    DM u_l_star = solution.value(Ul[0]);  
    /* Needs to be pointer assigned to the outside */
    // return {u_r_star, u_l_star};  
};
