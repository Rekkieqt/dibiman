#include <casadi/casadi.hpp>
#include "dibiman/nmpc/baseNMPC.hpp"
#include "dibiman/nmpc/centralizedNMPC.hpp"
#include "dibiman/utils/utils.hpp"

using namespace dibiman;

centralizedNMPC::centralizedNMPC(
        const std::string & model_path,
        const std::vector<std::vector<std::string>> & joints_to_use,
        const std::vector<std::string>& list_ee_frame_names,
        const std::vector<std::string>& ids
        )
{
    H = 50;
    dt = 0.05f;
    int i = 0;
    for (const auto& id : ids)
    {
        addArmToList(model_path, joints_to_use[i], list_ee_frame_names[i], id);
        i++;
    }

    na = armList[0].model.nv;
    nu = armList[0].model.njoints - 1;
    nx = 2 * na;

    verifyManipulatorList();

    createOCP();
}

void
centralizedNMPC::createOCP(void) {  
    using namespace casadi;  

    SX x0_r = SX::sym("x0_r", nx), xref_r = SX::sym("xref_r", nx), uref_r = SX::sym("uref_r", nu);  
    SX x0_l = SX::sym("x0_l", nx), xref_l = SX::sym("xref_l", nx), uref_l = SX::sym("uref_l", nu); 

    std::vector<SX> Xr, Ur, Ar, Xl, Ul, Al;
    for (int k = 0; k < H; ++k) {  
        Xr.push_back(SX::sym("Xr"+std::to_string(k), nx));
        Ur.push_back(SX::sym("Ur"+std::to_string(k), nu));
        Ar.push_back(SX::sym("Ar"+std::to_string(k), na));

        Xl.push_back(SX::sym("Xl"+std::to_string(k), nx));
        Ul.push_back(SX::sym("Ul"+std::to_string(k), nu));
        Al.push_back(SX::sym("Al"+std::to_string(k), na));
    }  

    Xr.push_back(SX::sym("Xr"+std::to_string(H+1), nx));
    Xl.push_back(SX::sym("Xl"+std::to_string(H+1), nx));

    DM Ru = 1 * DM::eye(nu);  
    DM Qx = 7 * DM::eye(nx);  
    DM Qa = DM::eye(na);  
  
    SX obj = 0;
    std::vector<SX> g; // Equality constraints
    g.push_back(Xr[0] - x0_r);
    g.push_back(Xl[0] - x0_l);

    // Dynamics + Jacobian binding loop  
    for (int k = 0; k < H; ++k) 
    {  
        SX xr_next = armList[0].inv_dyn_f(std::vector<SX>{Xr[k], Ar[k]}).at(0);  
        g.push_back(Xr[k+1] - xr_next);  
      
        SX ur_k = armList[0].rnea_h(std::vector<SX>{Xr[k], Ar[k]}).at(0);  
        g.push_back(Ur[k] - ur_k);  
      
        SX xl_next = armList[1].inv_dyn_f(std::vector<SX>{Xl[k], Al[k]}).at(0);  
        g.push_back(Xl[k+1] - xl_next);  
      
        SX ul_k = armList[1].rnea_h(std::vector<SX>{Xl[k], Al[k]}).at(0);  
        g.push_back(Ul[k] - ul_k);  
      
        // Hand-alignment constraint via jac_h  
        SX err = armList[1].jac_h(std::vector<SX>{Xr[k+1]}).at(0) - armList[1].jac_h(std::vector<SX>{Xl[k+1]}).at(0);  
        g.push_back(err);  
    }    

    for (int i = 0; i < H; ++i) 
    {
        obj += SX::mtimes({(Xr[i+1] - xref_r).T(), Qx, Xr[i+1] - xref_r});
        obj += SX::mtimes({(Ur[i]   - uref_r).T(), Ru,  Ur[i]   - uref_r});
        obj += SX::mtimes({Ar[i].T(), Qa, Ar[i]});
  
        obj += SX::mtimes({(Xl[i+1] - xref_l).T(), Qx, Xl[i+1] - xref_l});
        obj += SX::mtimes({(Ul[i]   - uref_l).T(), Ru,  Ul[i]   - uref_l});
        obj += SX::mtimes({Al[i].T(), Qa, Al[i]});
    }
  
    // SX X = vertcat(vertcat(Xr), vertcat(Xl), vertcat(Ur), vertcat(Ul), vertcat(Ar), vertcat(Al));  
    SX X = vertcat(vertcat(Xr), vertcat(Ur), vertcat(Ar), vertcat(Xl), vertcat(Ul), vertcat(Al));  

    /*
    0, (H+1)*nx = Xr
    (H+1)*nx, (H+1)*nx + H*nu = Ur
    (H+1)*nx + H*nu, (H+1)*nx + H*nu + H*na = Ar 
    (H+1)*nx + H*nu + H*na, (H+1)*nx + H*nu + H*na + (H+1)*nx = Xl
    (H+1)*nx + H*nu + H*na + (H+1)*nx, (H+1)*nx + H*nu + H*na + (H+1)*nx + H*nu = Ul
    (H+1)*nx + H*nu + H*na + (H+1)*nx + H*nu, (H+1)*nx + H*nu + H*na + (H+1)*nx + H*nu + H*na = Al
    */

    SX P = vertcat(x0_r, xref_r, uref_r, x0_l, xref_l, uref_l);
    SX G = vertcat(g);  
  
    SXDict nlp = {{"x", X}, {"f", obj}, {"g", G}, {"p", P}};  
  
    Dict solver_options;  
    solver_options["print_time"] = false;  
    solver_options["ipopt.print_level"] = 0;  
    solver_options["ipopt.tol"] = 1e-3;  
  
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

std::map<std::string, casadi::DM>
centralizedNMPC::solve(const std::map<std::string, armTargetParams> & manipulator_datas)
{  
    using namespace casadi;
    /*
       Eigen::VectorXd ex = ...;
       std::vector<double> temp(ex.data(), ex.data() + ex.size()) 
       DM ex_dm = DM(temp);
    */
    std::vector<DM> p_val_list;
    for (auto arm = armList.begin();
            arm != armList.end();
            ++arm)
    {
        const armTargetParams & m_data = manipulator_datas.at(arm->id);
        DM x0_dm = eigenToDM(m_data.x0);
        std::cout << arm->id << "\n" << x0_dm << "\n";
        DM xref_dm = eigenToDM(m_data.xref);
        DM uref_dm = eigenToDM(m_data.uref);
        p_val_list.push_back(x0_dm);
        p_val_list.push_back(xref_dm);
        p_val_list.push_back(uref_dm);
    }
  
    // DM x0 = DM::zeros(X.size1());
    casadi_int ng = solver.size1_in(solver.index_in("lbg"));
    DM lbg = DM::zeros(ng);
    DM ubg = DM::zeros(ng);

    DM p_val = vertcat(p_val_list);

    DMDict arg = {{"p", p_val}, {"lbg", lbg}, {"ubg", ubg}};
    DMDict res = solver(arg);

    DM x_opt = res.at("x");
    std::map<std::string, DM> u_star;

    /*
    0, (H+1)*nx = Xr
    (H+1)*nx, (H+1)*nx + H*nu = Ur
    (H+1)*nx + H*nu, (H+1)*nx + H*nu + H*na = Ar 
    (H+1)*nx + H*nu + H*na, (H+1)*nx + H*nu + H*na + (H+1)*nx = Xl
    (H+1)*nx + H*nu + H*na + (H+1)*nx, (H+1)*nx + H*nu + H*na + (H+1)*nx + H*nu = Ul
    (H+1)*nx + H*nu + H*na + (H+1)*nx + H*nu, (H+1)*nx + H*nu + H*na + (H+1)*nx + H*nu + H*na = Al
    */

    // needs to be redone
    casadi_int off_Ur = (H+1)*nx;
    casadi_int off_Ul = off_Ur; // Ur
    off_Ul += H*nu; // Ar
    off_Ul += H*na; // Xl
    off_Ul += (H+1)*nx; // Ul

    u_star[armList[0].id] = x_opt(Slice(off_Ur, off_Ur + nu));
    u_star[armList[1].id] = x_opt(Slice(off_Ul, off_Ul + nu));

    return u_star;
};
