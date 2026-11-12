//
// Copyright (c) 2017-2020 CNRS INRIA
//

#include "pinocchio/multibody.hpp"
#include <iostream>

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
  Eigen::Tensor<double, 0> sum_test = data.d2tau_dqdq.sum();
  std::cout << sum_test << std::endl;
  Eigen::Tensor<double, 3> mul_test = data.d2tau_dqdq * data.d2tau_dqdq.constant(0.2f);
  Eigen::Tensor<double, 1> lam(model.nv);
  lam.setRandom();
  // double lam_[model.nv];
  std::vector<double> lam_vec(model.nv);
  // Eigen::Tensor<double, 2> rnea_contract = lam * data.d2tau_dqdq;
  Eigen::array<Eigen::IndexPair<int>, 1> dims = { Eigen::IndexPair<int>(0, 0) };
  Eigen::Tensor<double, 2> rnea_result = lam.contract(data.d2tau_dqdq, dims);
  std::cout << rnea_result.size() << std::endl;
  std::cout << rnea_result.dimension(0) << std::endl;
  std::cout << rnea_result.dimension(1) << std::endl;
  Eigen::Map<Eigen::MatrixXd> rnea_mat(rnea_result.data(), rnea_result.dimension(0), rnea_result.dimension(1));
  Eigen::TensorMap<const Eigen::Tensor<double, 1>> lam_ten(lam_vec.data(), model.nv);
  Eigen::Tensor<double, 3> flat_test(2, 2, 2);

  flat_test(0, 0, 0) = 1;
  flat_test(0, 0, 1) = 2;
  flat_test(0, 1, 0) = 3;
  flat_test(0, 1, 1) = 4;

  flat_test(1, 0, 0) = 5;
  flat_test(1, 0, 1) = 6;
  flat_test(1, 1, 0) = 7;
  flat_test(1, 1, 1) = 8;


  double val = flat_test(1, 1, 1);
  Eigen::Map<Eigen::VectorXd> flat_test_vec(flat_test.data(), flat_test.size());
  std::cout << flat_test_vec.transpose() << std::endl;

}

