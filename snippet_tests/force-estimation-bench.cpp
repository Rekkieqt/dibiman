#include <limits>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <memory>

#include "dibiman/utils/robot.hpp"

#include "pinocchio/eigen-common.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include "pinocchio/algorithm/frames.hpp"

#include <iDynTree/ModelLoader.h>
#include <iDynTree/ExtWrenchesAndJointTorquesEstimator.h>
#include <iDynTree/Sensors.h>
#include <iDynTree/JointState.h>
#include <iDynTree/ExternalWrenchesEstimation.h>
#include <iDynTree/ContactWrench.h>
#include <iDynTree/Wrench.h>
#include <iDynTree/EigenHelpers.h>

int 
main()
{
    const std::string modelpath = static_cast<std::string>(std::filesystem::current_path()) + "/../../conf/model_FT.urdf";
    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};
    {
        using namespace pinocchio;
        Model model = dibiman::getArmModel(modelpath, joint_list);
        Data data = Data(model);

        Force::Vector6 f_vec;
        f_vec << -7.44204, -3.78867, 15.6993, 0.278561, -1.57442, -0.0548361;
        Force f_measured(f_vec);
        
        // Sensor frame should exist in the model (e.g. FrameType::SENSOR from URDF,  
        // or an OP_FRAME you added at the sensor location)  
        const FrameIndex sensor_id = model.getFrameId("r_arm_ft_sensor");  

        Eigen::VectorXd q = neutral(model);
        for (int i = 0; i < 5; ++i)
        {
            auto t0 = std::chrono::steady_clock::now();  

            // With the robot static and unloaded: v = a = 0 → only gravity contributes
            pinocchio::rnea(model, data, q, Eigen::VectorXd::Zero(model.nv), Eigen::VectorXd::Zero(model.nv));  
            // Model-predicted wrench at the sensor (LOCAL frame)  
            const Force f_model = pinocchio::computeSupportedForceByFrame(model, data, sensor_id);  
            // Offset = what the sensor reads minus what the model predicts  
            const Force f_offset = f_measured - f_model;   // both in sensor-local frame

            auto t1 = std::chrono::steady_clock::now();  

            auto t1_eval = std::chrono::duration_cast<std::chrono::microseconds> (t1 - t0).count();

            std::cout << "t1_eval :"  << t1_eval << "\n";
            std::cout << "f_model :"  << f_model << "\n";
        }
    }

    {
        using namespace iDynTree;

        // 1) Load the estimator and model/sensors
        iDynTree::ExtWrenchesAndJointTorquesEstimator estimator;

        if (!estimator.loadModelAndSensorsFromFile(modelpath)) {
            std::cerr << "Failed to load model and sensors from " << modelpath << std::endl;
            return 1;
        }

        // Print model string (equivalent to toString())
        std::cout << estimator.model().toString() << std::endl;

        // 2) Set kinematics information
        // Assume robot is balancing on left foot, using l_sole frame as fixed base
        iDynTree::Vector3 gravity;
        gravity(0) = 0.0;
        gravity(1) = 0.0;
        gravity(2) = -9.81;

        // Get joint information: all zeros for this example
        size_t dofs = estimator.model().getNrOfDOFs();
        
        iDynTree::JointPosDoubleArray qj_idyn(dofs);
        iDynTree::JointDOFsDoubleArray dqj_idyn(dofs);
        iDynTree::JointDOFsDoubleArray ddqj_idyn(dofs);
        
        qj_idyn.zero();
        dqj_idyn.zero();
        ddqj_idyn.zero();

        // Set the kinematics information in the estimator
        iDynTree::FrameIndex l_sole_index = estimator.model().getFrameIndex("r_hand_dh_frame");
        if (l_sole_index == iDynTree::FRAME_INVALID_INDEX) {
            std::cerr << "Frame 'l_sole' not found in model." << std::endl;
            return 1;
        }
        
        if (!estimator.updateKinematicsFromFixedBase(qj_idyn, dqj_idyn, ddqj_idyn, l_sole_index, gravity)) {
            std::cerr << "Failed to update kinematics from fixed base." << std::endl;
            return 1;
        }

        // 3) Specify unknown wrenches
        // We assume an unknown full wrench at the origin of the l_sole frame
        iDynTree::UnknownWrenchContact unknownWrench;
        unknownWrench.unknownType = iDynTree::FULL_WRENCH;
        unknownWrench.contactPoint.zero();

        // Store all unknown external wrenches
        iDynTree::LinkUnknownWrenchContacts fullBodyUnknowns(estimator.model());
        fullBodyUnknowns.clear();
        fullBodyUnknowns.addNewContactInFrame(estimator.model(), l_sole_index, unknownWrench);

        // Print the unknowns for verification
        std::cout << fullBodyUnknowns.toString(estimator.model()) << std::endl;

        // 4) Run the prediction of FT measurements
        iDynTree::SensorsMeasurements estFTmeasurements(estimator.sensors());
        iDynTree::JointDOFsDoubleArray estJointTorques(dofs);
        iDynTree::LinkContactWrenches estContactForces(estimator.model());

        // Run the estimation
        if (!estimator.computeExpectedFTSensorsMeasurements(fullBodyUnknowns, estFTmeasurements, estContactForces, estJointTorques)) {
            std::cerr << "computeExpectedFTSensorsMeasurements failed." << std::endl;
            return 1;
        }

        // 5) Print the estimated measurements
        // Get the number of FT sensors
        int nrOfFTSensors = estimator.sensors().getNrOfSensors(iDynTree::SIX_AXIS_FORCE_TORQUE);
        
        for (int ftIndex = 0; ftIndex < nrOfFTSensors; ++ftIndex) {
            iDynTree::Wrench estimatedSensorWrench;
            iDynTree::Sensor* sens = estimator.sensors().getSensor(iDynTree::SIX_AXIS_FORCE_TORQUE, ftIndex);
            
            auto t0 = std::chrono::steady_clock::now();  

            estimator.updateKinematicsFromFixedBase(qj_idyn, dqj_idyn, ddqj_idyn, l_sole_index, gravity);

            estimator.computeExpectedFTSensorsMeasurements(fullBodyUnknowns, estFTmeasurements, estContactForces, estJointTorques);

            estFTmeasurements.getMeasurement(iDynTree::SIX_AXIS_FORCE_TORQUE, ftIndex, estimatedSensorWrench);

            auto t1 = std::chrono::steady_clock::now();  

            auto t1_eval = std::chrono::duration_cast<std::chrono::microseconds> (t1 - t0).count();

            std::cout << "t2_eval :"  << t1_eval << "\n";

            std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            std::cout << "Sensor " << sens->getName() << " has index " << ftIndex << std::endl;
            std::cout << "Estimated measured wrench: " << estimatedSensorWrench.toString() << std::endl;
        }

        // Print the estimated contact forces
        std::cout << estContactForces.toString(estimator.model()) << std::endl;
    }
    return 0;
}
