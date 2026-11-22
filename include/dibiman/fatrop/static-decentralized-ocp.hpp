#pragma once

#include <fatrop/fatrop.hpp>
#include "dibiman/fatrop/nlp_ocp.hxx"
#include "dibiman/fatrop/decentralized-fatrop-helper.hpp"
#include "dibiman/utils/robot.hpp"

namespace dibiman 
{
    struct DecentralizedManipulatorStatic {};
    typedef fatrop::OcpAbstractTpl<DecentralizedManipulatorStatic> DecentralizedOcp;
}

namespace fatrop {

template <> 
class OcpAbstractTpl<dibiman::DecentralizedManipulatorStatic> : public dibiman::DecentralizedFatropHelper
{
    public:
        using Index = fatrop::Index;
        using Scalar = fatrop::Scalar;

        OcpAbstractTpl(const pinocchio::Model &, const std::string &, const int &, const int &);

        Index get_nx(const Index k) const;
        Index get_nu(const Index k) const;
        Index get_ng(const Index k) const;
        Index get_ng_ineq(const Index k) const;
        Index get_horizon_length() const;
        Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k);
        Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                          const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k,
                          MAT *res, const Index k);
        Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k);
        Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k,
                     Scalar *res, const Index k);
        Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k);
        Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                      Scalar *res, const Index k);
        Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k,
                     Scalar *res, const Index k);
        Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k);
        Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k);
        Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const;
        Index get_initial_xk(Scalar *xk, const Index k) const;
        Index get_initial_uk(Scalar *uk, const Index k) const;
};

}
