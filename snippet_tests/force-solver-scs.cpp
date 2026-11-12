#include <iostream>

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "scs.h"

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

}


