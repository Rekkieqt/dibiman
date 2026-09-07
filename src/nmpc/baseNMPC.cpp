#include "dibiman/nmpc/baseNMPC.hpp"
#include <Eigen/Dense>

#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
#include "pinocchio/algorithm/rnea.hpp"  
#include "pinocchio/algorithm/frames.hpp"  
#include "pinocchio/algorithm/model.hpp"  

#include "pinocchio/parsers/urdf.hpp"  

#include "pinocchio/autodiff/casadi.hpp"
#include "pinocchio/math.hpp"

#include "pinocchio/utils/cast.hpp"
#include "pinocchio/utils/check.hpp"

#include <casadi/casadi.hpp>
  
using namespace dibiman;

pinocchio::Model::ConfigVectorType 
baseNMPC::inverseKinematics(  
    const pinocchio::Model & model,  
    const pinocchio::Model::ConfigVectorType & q0,  
    const pinocchio::SE3 & Href,  
    const pinocchio::FrameIndex frame_id,  
    const std::string & target = "full")  
{  
    using namespace pinocchio;
    Data data(model);  
    const double eps = 1e-6;  
    const int IT_MAX = 4000;  
    const double DT = 1e-1;  
    const double damp = 1e-12;  
    
    Eigen::VectorXd q = q0;  
    Eigen::VectorXd err(target == "full" ? 6 : 3);  
    Eigen::MatrixXd J(6, model.nv);  
    bool success = false;  
    int i = 0;  
    
    for (;;)  
    {  
        forwardKinematics(model, data, q);  
        updateFramePlacement(model, data, frame_id);  
    
      if (target == "full")  
      {  
          const SE3 iMd = data.oMf[frame_id].actInv(Href);  
          err = pinocchio::log6(iMd).toVector();  
        
          J.setZero();  
          computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL, J);  
          J = -Jlog6(iMd.inverse()) * J;  
        
          Eigen::VectorXd v = -J.transpose() * (J * J.transpose() + damp * Eigen::MatrixXd::Identity(6, 6))  
                                   .ldlt().solve(err);  
          q = integrate(model, q, v * DT);  
      }  
      else  
      {  
          err = data.oMf[frame_id].translation() - Href.translation();  
        
          Eigen::MatrixXd J6(6, model.nv);  
          J6.setZero();  
          computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6);  
          Eigen::MatrixXd J3 = J6.topRows<3>();  
        
          Eigen::VectorXd v = -J3.transpose() * (J3 * J3.transpose() + damp * Eigen::MatrixXd::Identity(3, 3))  
                                   .ldlt().solve(err);  
          q = integrate(model, q, v * DT);  
      }  
    
      if (err.norm() < eps) { success = true; break; }  
      if (i >= IT_MAX) { success = false; break; }  
      ++i;  
    }  
    
    if (success)  
        std::cout << "Convergence achieved for " << model.frames[frame_id].name << "!" << std::endl;  
    else  
        std::cout << "\nWarning: the iterative algorithm has not reached convergence to the desired precision "  
                 << model.frames[frame_id].name << std::endl;  
    
    return q;  
};
  
void
baseNMPC::getArmModel(const std::string & modelpath,  
                      const std::vector<std::string> & joints_to_use,
                      const std::string & prefix,
                      icubArm _arm)  
{  
    using namespace pinocchio;
    std::vector<std::string> jointsToFree;  
    for (const auto & jnt : joints_to_use) jointsToFree.push_back(prefix + jnt);  

    jointsToFree.push_back("universe");  

    Model fullModel;  
    pinocchio::urdf::buildModel(modelpath, fullModel);  

    std::vector<JointIndex> jointsToLockIDs;  
    for (const auto & jn : fullModel.names)  
    {  
    if (std::find(jointsToFree.begin(), jointsToFree.end(), jn) == jointsToFree.end())  
        jointsToLockIDs.push_back(fullModel.getJointId(jn));  
    }  

    Eigen::VectorXd initConf = Eigen::VectorXd::Zero(fullModel.nq);  

    Model model = buildReducedModel(fullModel, jointsToLockIDs, initConf);  
    Data data(model);  

    _arm.model = model;
    _arm.data = data;
    _arm.hand_id = model.getFrameId(prefix + "hand");
    na = model.nv;
    nu = model.njoints - 1;
    nx = na * 2;
};

void
baseNMPC::inverseModel(icubArm _arm) 
{
    using namespace casadi;  
  
    // Dynamic variables  
    SX q = SX::sym("q", nu);  
    SX v = SX::sym("v", nu);  
    SX a = SX::sym("a", na);  
  
    // Simplified dynamics  
    SX x = vertcat(q, v);  
    SX dx = vertcat(v, a);  
  
    SX qk = q + dt * v;  
    SX vk = v + dt * a;  
    SX xk = vertcat(qk, vk);  
  
    _arm.inv_dyn_f = Function("Fk", {x, a}, {xk}, {"x", "a"}, {"xk"}).expand();  
};

void
baseNMPC::rnea(icubArm _arm)
{  
    using namespace pinocchio;
    typedef ::casadi::SX ADScalar;  
    
    typedef pinocchio::ModelTpl<ADScalar> ADModel;  
    typedef ADModel::Data ADData;  
    
    ADModel ad_model = _arm.model.cast<ADScalar>();  
    ADData ad_data(ad_model);  
    
    typedef ADModel::ConfigVectorType ConfigVectorAD;  
    typedef ADModel::TangentVectorType TangentVectorAD;  
    
    ::casadi::SX cs_q = ::casadi::SX::sym("q", na);  
    ConfigVectorAD q_ad(na);  
    q_ad = Eigen::Map<ConfigVectorAD>(  
      static_cast<std::vector<ADScalar>>(cs_q).data(), na, 1);  
    
    ::casadi::SX cs_v = ::casadi::SX::sym("v", na);  
    TangentVectorAD v_ad(na);  
    v_ad = Eigen::Map<TangentVectorAD>(  
      static_cast<std::vector<ADScalar>>(cs_v).data(), na, 1);  
    
    ::casadi::SX cs_a = ::casadi::SX::sym("a", na);  
    TangentVectorAD a_ad(na);  
    a_ad = Eigen::Map<TangentVectorAD>(  
      static_cast<std::vector<ADScalar>>(cs_a).data(), na, 1);  
    
    pinocchio::rnea(ad_model, ad_data, q_ad, v_ad, a_ad);  
    
    ::casadi::SX tau_ad(na, 1);  
    for (Eigen::Index k = 0; k < na; ++k)  
      tau_ad(k) = ad_data.tau[k];  
    
    _arm.rnea_h = ::casadi::Function(  
      "eval_rnea", ::casadi::SXVector{cs_q, cs_v, cs_a}, ::casadi::SXVector{tau_ad});  

    /*
    Function rneaJac("jac_rnea",  
        {x, a, MX(1,1)==SX(1,1) ? SX(1,1) : SX(1,1)}, // placeholder for tau output  
        {horzcat(du_dq, du_dv), du_da},  
        {"x", "a", "out_tau"},  
        {"jac_tau_x", "jac_tau_a"});  
  
    _arm.rnea = Function("rnea", {x, a}, {tau}, {"x", "a"}, {"tau"},  
        Dict{{"custom_jacobian", rneaJac}, {"jac_penalty", 0}}).expand();  
        */
};

void
baseNMPC::armJacobian(icubArm _arm)
{
    typedef pinocchio::ModelTpl<casadi::SX> CasadiModel;  
    typedef pinocchio::DataTpl<casadi::SX> CasadiData;  
      
    CasadiModel cmodel = _arm.model.cast<casadi::SX>();  
    CasadiData cdata(cmodel);  
      
    ::casadi::SX q_sx = casadi::SX::sym("q", na);  
    ::casadi::SX v_sx = casadi::SX::sym("v", na);  
      
    Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1> q =  
      Eigen::Map<Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>>(  
        static_cast<std::vector<casadi::SX>>(q_sx).data(), na);  
    Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1> v =  
      Eigen::Map<Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>>(  
        static_cast<std::vector<casadi::SX>>(v_sx).data(), na);  
      
    CasadiData::Matrix6x J(6, na);  
    J.setZero();  
    pinocchio::computeFrameJacobian(cmodel, cdata, q, _arm.hand_id, pinocchio::WORLD, J);  
      
    ::casadi::SX J_sx;  
    pinocchio::casadi::copy(J, J_sx);   // Eigen<SX> -> casadi::SX  
      
    ::casadi::SX dx = ::casadi::SX::mtimes(J_sx, v_sx);  
    ::casadi::SX x = ::casadi::SX::vertcat({q_sx, v_sx});  
      
    _arm.jac_h = ::casadi::Function("hand_jac", {x}, {dx}, {"x"}, {"spatial_vel"});
};

void
baseNMPC::getNewData(void)
{
    using namespace pinocchio;
    for (auto& _arm : armList) 
    {
        _arm.data = Data(_arm.model);
    }
};

void
baseNMPC::addArmToList(
        const std::string & modelpath, 
        const std::vector<std::string> & joints_to_use, 
        const std::string & prefix
        )
{
    icubArm _arm;
    _arm.prefix = prefix;
    getArmModel(modelpath, joints_to_use, prefix, _arm);
    rnea(_arm);
    // forwardModel(_arm);
    inverseModel(_arm);
    armJacobian(_arm);

    armList.push_back(_arm);
};
/*
void
baseNMPC::forwardModel(icubArm _arm)
{
    using namespace casadi;  
  
    SX u = SX::sym("tau", cmodel.nv);  
    SX q = SX::sym("q", cmodel.nq);  
    SX v = SX::sym("v", cmodel.nv);  
  
    // ABA  
    SX ddq = cpin::aba(cmodel, cdata, q, v, u);  
    SX x = vertcat(q, v)
  
    Function dx_f = Function("dx_f", {x, u}, {dx}, {"x", "u"}, {"dx"})
  
    SX xk = x + dt * dx_f(SXVector{x, u}).at(0);  
  
    _arm.for_dyn_f = Function("Fk", {x, u}, {xk}, {"x0", "u"}, {"xf"}).expand();  
};
*/
