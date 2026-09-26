#include "dibiman/nmpc/baseNMPC.hpp"
#include "dibiman/utils/robot.hpp"
#include <Eigen/Dense>

#include "pinocchio/algorithm/joint-configuration.hpp"  
#include "pinocchio/algorithm/kinematics.hpp"  
#include "pinocchio/algorithm/jacobian.hpp"  
#include "pinocchio/algorithm/rnea.hpp"  
#include "pinocchio/algorithm/frames.hpp"  
#include "pinocchio/algorithm/model.hpp"  
#include "pinocchio/algorithm/default-check.hpp"

#include "pinocchio/parsers/urdf.hpp"  

#include "pinocchio/autodiff/casadi.hpp"
#include "pinocchio/math.hpp"

#include "pinocchio/utils/cast.hpp"
#include "pinocchio/utils/check.hpp"

#include <casadi/casadi.hpp>
  
using namespace dibiman;

casadi::Function
baseNMPC::inverseModel(const int n_dim)
{
    using namespace casadi;
  
    // Dynamic variables  
    SX q = SX::sym("q", n_dim);
    SX v = SX::sym("v", n_dim);
    SX a = SX::sym("a", n_dim);
  
    SX x = vertcat(q, v);
    SX dx = vertcat(v, a);
  
    SX qk = q + dt * v;
    SX vk = v + dt * a;
    SX xk = vertcat(qk, vk);
  
    return Function("simple_dyn", {x, a}, {xk}, {"x", "a"}, {"xk"}).expand();
};

casadi::Function
baseNMPC::rnea(const icubArm & _arm)
{  
    using namespace pinocchio;
    typedef ::casadi::SX ADScalar;
    
    typedef pinocchio::ModelTpl<ADScalar> ADModel;  
    typedef ADModel::Data ADData;  
    
    ADModel ad_model = _arm.model.cast<ADScalar>();  
    ADData ad_data(ad_model);  
    
    typedef ADModel::ConfigVectorType ConfigVectorAD;  
    typedef ADModel::TangentVectorType TangentVectorAD;  
    
    ::casadi::SX cs_q = ::casadi::SX::sym("q", _arm.model.nv);  
    ConfigVectorAD q_ad(_arm.model.nv);  
    q_ad = Eigen::Map<ConfigVectorAD>(  
      static_cast<std::vector<ADScalar>>(cs_q).data(), _arm.model.nv, 1);  
    
    ::casadi::SX cs_v = ::casadi::SX::sym("v", _arm.model.nv);  
    TangentVectorAD v_ad(_arm.model.nv);
    v_ad = Eigen::Map<TangentVectorAD>(  
      static_cast<std::vector<ADScalar>>(cs_v).data(), _arm.model.nv, 1);  
    
    ::casadi::SX cs_a = ::casadi::SX::sym("a", _arm.model.nv);  
    TangentVectorAD a_ad(_arm.model.nv);  
    a_ad = Eigen::Map<TangentVectorAD>(  
      static_cast<std::vector<ADScalar>>(cs_a).data(), _arm.model.nv, 1);  
    
    pinocchio::rnea(ad_model, ad_data, q_ad, v_ad, a_ad);  
    
    ::casadi::SX tau_ad(_arm.model.nv, 1);  
    for (Eigen::Index k = 0; k < _arm.model.nv; ++k)  
      tau_ad(k) = ad_data.tau[k];  
    
    ::casadi::SX cs_x = ::casadi::SX::vertcat({cs_q, cs_v});
    return ::casadi::Function("rnea", ::casadi::SXVector{cs_x, cs_a}, ::casadi::SXVector{tau_ad});  
    //return ::casadi::Function("rnea", ::casadi::SXVector{cs_q, cs_v, cs_a}, ::casadi::SXVector{tau_ad});  

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

casadi::Function
baseNMPC::armJacobian(const icubArm & _arm)
{
    typedef pinocchio::ModelTpl<casadi::SX> CasadiModel;  
    typedef pinocchio::DataTpl<casadi::SX> CasadiData;  
      
    CasadiModel cmodel = _arm.model.cast<casadi::SX>();  
    CasadiData cdata(cmodel);  
      
    ::casadi::SX q_sx = casadi::SX::sym("q", _arm.model.nv);  
    ::casadi::SX v_sx = casadi::SX::sym("v", _arm.model.nv);  
      
    Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1> q =  
      Eigen::Map<Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>>(  
        static_cast<std::vector<casadi::SX>>(q_sx).data(), _arm.model.nv);  
    Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1> v =  
      Eigen::Map<Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>>(  
        static_cast<std::vector<casadi::SX>>(v_sx).data(), _arm.model.nv);  
      
    CasadiData::Matrix6x J(6, _arm.model.nv);  
    J.setZero();  
    pinocchio::computeFrameJacobian(cmodel, cdata, q, _arm.hand_id, pinocchio::WORLD, J);  
      
    ::casadi::SX J_sx;  
    pinocchio::casadi::copy(J, J_sx);   // Eigen<SX> -> casadi::SX  
      
    ::casadi::SX dx = ::casadi::SX::mtimes(J_sx, v_sx);  
    ::casadi::SX x = ::casadi::SX::vertcat({q_sx, v_sx});  
      
    return ::casadi::Function("jac_x_vel", {x}, {dx}, {"x"}, {"spatial_vel"});
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
        const std::string & name_ee_frame,
        const std::string & id
        )
{
    icubArm _arm;

    _arm.id = id;
    _arm.hand_name = name_ee_frame;
    _arm.model = getArmModel(modelpath, joints_to_use);
    _arm.data = pinocchio::Data(_arm.model);
    _arm.hand_id = _arm.model.getFrameId(name_ee_frame);

    _arm.rnea_h = rnea(_arm);
    _arm.for_dyn_f = forwardModel(_arm);
    _arm.inv_dyn_f = inverseModel(_arm.model.nv);
    _arm.jac_h = armJacobian(_arm);

    armList.push_back(_arm);
};

void
baseNMPC::verifyManipulatorList(void)
{
    for (const auto & _arm : armList)
    {
        std::cout << "List manipulator identifier: " << _arm.id << "\n";
        std::cout << "Hand pinocchio id: " << _arm.hand_id << std::endl;

        std::cout << _arm.model.check(pinocchio::DEFAULT_CHECKERS);
        std::cout << _arm.model.check(_arm.data) << std::endl;

        auto printFunctionInfo = [](const std::string& label, const casadi::Function& f) {  
            if (f.is_null()) {  
                std::cout << label << ": NULL (not initialized)\n";  
                return;  
            }  
            std::cout << label << ": " << f.name()  
                       << " (n_in=" << f.n_in() << ", n_out=" << f.n_out() << ")\n";  
            for (casadi_int i = 0; i < f.n_in(); ++i) {  
                std::cout << "  in[" << i << "] " << f.name_in(i) << ": "  
                           << f.size1_in(i) << "x" << f.size2_in(i) << "\n";  
            }  
            for (casadi_int i = 0; i < f.n_out(); ++i) {  
                std::cout << "  out[" << i << "] " << f.name_out(i) << ": "  
                           << f.size1_out(i) << "x" << f.size2_out(i) << "\n";  
            }  
        };
          
        printFunctionInfo("Casadi Inverse Dynamics", _arm.inv_dyn_f);  
        printFunctionInfo("Casadi RNEA", _arm.rnea_h);  
        printFunctionInfo("Casadi Discrete ABA", _arm.for_dyn_f);
        printFunctionInfo("Casadi Spatial Velocity (J(q) * v)", _arm.jac_h);

        std::cout << "u dim: " << nu << "a dim: " << na << "x dim: " << nx << std::endl;
    }
}

casadi::Function
baseNMPC::forwardModel(const icubArm & _arm)
{
    using namespace pinocchio;

    typedef double Scalar;
    typedef ::casadi::SX ADScalar;

    typedef ModelTpl<Scalar> Model;

    typedef ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;

    const Model & model = _arm.model;

    // Pick up random configuration, velocity and acceleration vectors.
    Eigen::VectorXd q(model.nq);
    q = randomConfiguration(model);
    Eigen::VectorXd v(Eigen::VectorXd::Random(model.nv));
    Eigen::VectorXd tau(Eigen::VectorXd::Random(model.nv));

    // Create CasADi model and data from model
    typedef ADModel::ConfigVectorType ConfigVectorAD;
    typedef ADModel::TangentVectorType TangentVectorAD;
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);

    // Create symbolic CasADi vectors
    ::casadi::SX cs_q = ::casadi::SX::sym("q", model.nq);
    ConfigVectorAD q_ad(model.nq);
    q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADScalar>>(cs_q).data(), model.nq, 1);

    ::casadi::SX cs_v = ::casadi::SX::sym("v", model.nv);
    TangentVectorAD v_ad(model.nv);
    v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(cs_v).data(), model.nv, 1);

    ::casadi::SX cs_tau = ::casadi::SX::sym("tau", model.nv);
    TangentVectorAD tau_ad(model.nv);
    tau_ad =
    Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(cs_tau).data(), model.nv, 1);

    // Build CasADi function
    aba(ad_model, ad_data, q_ad, v_ad, tau_ad);
    ::casadi::SX a_ad(model.nv, 1);

    for (Eigen::Index k = 0; k < model.nv; ++k)
    a_ad(k) = ad_data.ddq[k];

    // ::casadi::Function eval_aba("eval_aba", ::casadi::SXVector{cs_q, cs_v, cs_tau}, ::casadi::SXVector{a_ad});
    ::casadi::SX cs_x = ::casadi::SX::vertcat({cs_q, cs_v});
    ::casadi::Function eval_aba("aba", ::casadi::SXVector{cs_x, cs_tau}, ::casadi::SXVector{a_ad});

    ::casadi::SX dx = ::casadi::SX::vertcat({cs_v, a_ad});

    ::casadi::SXDict dae = {{"x", cs_x}, {"p", cs_tau}, {"ode", dx}};

    ::casadi::Dict integrator_opts;
    integrator_opts["simplify"] = true;
    integrator_opts["number_of_finite_elements"] = 1;
    
    ::casadi::Function discr_aba = ::casadi::integrator("discr_aba", "rk", dae, 0, dt, integrator_opts);
    
    /* Discretization of the model */

    return discr_aba;
};

void
baseNMPC::setObjectFrame(const pinocchio::SE3 & objFrame, const std::map<std::string, Eigen::VectorXd> & jointData)
{
    for (auto & _arm : armList) 
    {
        const int _last_joint = _arm.model.njoints - 1;
        const pinocchio::SE3 hand_to_object_transform = _arm.data.oMi[_last_joint].inverse() * objFrame;

        const std::string _object_name = _arm.id + "_object";
        pinocchio::Frame _hand_to_object_frame(
                _object_name,
                _arm.model.frames[_last_joint].parentJoint,
                _arm.hand_id,
                hand_to_object_transform,
                pinocchio::FrameType::OP_FRAME);
                
        _arm.model.addFrame(_hand_to_object_frame);
        _arm.data = pinocchio::Data(_arm.model);
        const Eigen::VectorXd & q = jointData.at(_arm.id);
        pinocchio::forwardKinematics(_arm.model, _arm.data, q);
        pinocchio::updateFramePlacements(_arm.model, _arm.data);
        _arm.object_id = _arm.model.getFrameId(_object_name);
        _arm.hand_id = _arm.model.getFrameId(_arm.hand_name);
    }
}
