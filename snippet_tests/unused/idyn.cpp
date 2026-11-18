#include <iostream>
#include <iDynTree/ModelLoader.h>
#include <iDynTree/ExtWrenchesAndJointTorquesEstimator.h>
#include <iDynTree/Sensors.h>
#include <iDynTree/JointState.h>
#include <iDynTree/ExternalWrenchesEstimation.h>
#include <iDynTree/ContactWrench.h>
#include <iDynTree/Wrench.h>
#include <iDynTree/Vector3.h>

int main()
{
    // 1) Load the estimator and model/sensors
    iDynTree::ExtWrenchesAndJointTorquesEstimator estimator;
    std::string modelPath = "./iCubGenova02.urdf"; // Adjust path as needed
    
    if (!estimator.loadModelAndSensorsFromFile(modelPath)) {
        std::cerr << "Failed to load model and sensors from " << modelPath << std::endl;
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
    iDynTree::FrameIndex l_sole_index = estimator.model().getFrameIndex("l_sole");
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
        
        estFTmeasurements.getMeasurement(iDynTree::SIX_AXIS_FORCE_TORQUE, ftIndex, estimatedSensorWrench);

        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        std::cout << "Sensor " << sens->getName() << " has index " << ftIndex << std::endl;
        std::cout << "Estimated measured wrench: " << estimatedSensorWrench.toString() << std::endl;
    }

    // Print the estimated contact forces
    std::cout << estContactForces.toString(estimator.model()) << std::endl;

    // 6) Specific request: wrench measured at r_hand_dh_frame
    // Find the index of the "r_hand_ft_sensor"
    iDynTree::SensorsList sensors = estimator.sensors();
    std::ptrdiff_t rHandFtIndex = sensors.getSensorIndex(iDynTree::SIX_AXIS_FORCE_TORQUE, "r_hand_ft_sensor");
    
    if (rHandFtIndex != -1) {
        iDynTree::Wrench rHandFtWrench;
        // Note: The estimated measurement is already the wrench *measured by the sensor*,
        // which is associated with a specific frame in the URDF (typically the sensor frame).
        // The requested "r_hand_dh_frame" is likely the frame in which the sensor is mounted or expressed.
        // If you need the wrench expressed in a different frame, you would need to apply a frame transformation.
        // However, the sensor measurement itself is typically reported in its own frame or a designated child frame.
        estFTmeasurements.getMeasurement(iDynTree::SIX_AXIS_FORCE_TORQUE, rHandFtIndex, rHandFtWrench);
        std::cout << "Wrench measured by r_hand_ft_sensor: " << rHandFtWrench.toString() << std::endl;
        
        // If you need the wrench at "r_hand_dh_frame" specifically, you would need to:
        // 1. Get the transform between the sensor frame and r_hand_dh_frame
        // 2. Transform the wrench using the adjoint matrix (often done via iDynTree::Wrench::changeFrame)
        // This part is model-specific and depends on your URDF structure.
    } else {
        std::cerr << "Sensor 'r_hand_ft_sensor' not found." << std::endl;
    }

    return 0;
}
