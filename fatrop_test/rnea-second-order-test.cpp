//
// Copyright (c) 2017-2020 CNRS INRIA
//

#include "pinocchio/multibody.hpp"

#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/multibody/sample-models.hpp"


int
main()
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model);

  Data data(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill(1.);

  VectorXd q = randomConfiguration(model);
  VectorXd v(VectorXd::Random(model.nv));
  VectorXd a(VectorXd::Random(model.nv));

  // check with only q non-zero
  Data::Tensor3x dtau2_dq(model.nv, model.nv, model.nv);
  Data::Tensor3x dtau2_dv(model.nv, model.nv, model.nv);
  Data::Tensor3x dtau2_dqdv(model.nv, model.nv, model.nv);
  Data::Tensor3x dtau2_dadq(model.nv, model.nv, model.nv);
  dtau2_dq.setZero();
  dtau2_dv.setZero();
  dtau2_dqdv.setZero();
  dtau2_dadq.setZero();

  ComputeRNEASecondOrderDerivatives(model, data, q, VectorXd::Zero(model.nv), VectorXd::Zero(model.nv));

}

