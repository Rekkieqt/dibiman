/* yarp libraries */
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/ITorqueControl.h>
#include <yarp/dev/MultipleAnalogSensorsInterfaces.h>
#include <yarp/dev/IControlMode.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/IAxisInfo.h>

/* cpp libraries */
#include <vector>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <list>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <filesystem>

/* Pinocchio libraries */
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"

/* Casadi and Eigen */
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

/* NMPC libraries */
#include "dibiman/nmpc/centralizedNMPC.hpp"
#include "dibiman/nmpc/baseNMPC.hpp"
#include "dibiman/utils/robot.hpp"

/* Gazebo libraries */
#include <gz/transport/Node.hh>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/sim/Util.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>

using namespace yarp::os;
using namespace yarp::dev;
using namespace pinocchio;

void worldControl(uint64_t _steps, const std::string & _world_name, bool _paused=true)
{
    using namespace gz;
    msgs::WorldControl req;  
    req.set_pause(_paused);  
    req.set_multi_step(_steps);  

    msgs::Boolean rep;
    bool result = false;
    transport::Node node;  
    bool executed = node.Request("/world/" + _world_name + "/control", req, 2000, rep, result); 

    if (!(executed && result && rep.data()))
        std::cerr << "Failed to send request!\n";
}

int main(int argc, char **argv)
{
    /* __________________________________________ YARP NETWORKING __________________________________________ */

    /* findGroup -> find a list and save it as a Bottle, useful for later */
    const std::string right_robotName = "icubSim"; /* icubSim */
    const std::string right_partName = "right_arm"; /* {ARM}_arm */
    const std::string right_local = "/right_arm_controller"; /* /{ARM}_arm_controller */

    const std::string left_robotName = "icubSim"; /* icubSim */
    const std::string left_partName = "left_arm"; /* {ARM}_arm */
    const std::string left_local = "/left_arm_controller"; /* /{ARM}_arm_controller */

    /* __________________________________________ Right Arm __________________________________________ */

    /* configuring in and out ports */
    /* where to connect */
    std::string right_remotePorts="/";
    right_remotePorts+=right_robotName;
    right_remotePorts+="/";
    right_remotePorts+=right_partName;

    /* name of local port */
    std::string right_localPorts = "/right_arm_controller"; /* /{ARM}_arm_controller */

    /* connect to simulated control board */
    Property right_options;
    right_options.put("device", "remote_controlboard");
    right_options.put("local", right_localPorts.c_str());   
    right_options.put("remote", right_remotePorts.c_str()); 

    /* create a device */
    PolyDriver right_robotDevice(right_options);
    if (!right_robotDevice.isValid()) {
        yError("(right) Device not available.\n");
        return 0;
    }

    /* create interfaces */
    IControlMode *right_controlMode;
    ITorqueControl *right_torqueControl;
    IEncoders *right_encoders;
    IAxisInfo *right_axInfo;

    /* check if interfaces are available */
    bool ok;
    ok = right_robotDevice.view(right_controlMode);
    ok = ok && right_robotDevice.view(right_torqueControl);
    ok = ok && right_robotDevice.view(right_encoders);
    ok = ok && right_robotDevice.view(right_axInfo);

    if (!ok) {
        yError("Problems acquiring interfaces\n");
        return 0;
    }

    /* __________________________________________ Left Arm __________________________________________ */

    /* configuring in and out ports */
    /* where to connect */
    std::string left_remotePorts="/";
    left_remotePorts+=left_robotName;
    left_remotePorts+="/";
    left_remotePorts+=left_partName;

    /* name of local port */
    std::string left_localPorts = "/left_arm_controller"; /* /{ARM}_arm_controller */

    /* connect to simulated control board */
    Property left_options;
    left_options.put("device", "remote_controlboard");
    left_options.put("local", left_localPorts.c_str());   
    left_options.put("remote", left_remotePorts.c_str()); 

    /* create a device */
    PolyDriver left_robotDevice(left_options);
    if (!left_robotDevice.isValid()) {
        yError("(left) Device not available.\n");
        return 0;
    }

    /* create interfaces */
    IControlMode *left_controlMode;
    ITorqueControl *left_torqueControl;
    IEncoders *left_encoders;
    IAxisInfo *left_axInfo;

    /* check if interfaces are available */
    ok = false;
    ok = left_robotDevice.view(left_controlMode);
    ok = ok && left_robotDevice.view(left_torqueControl);
    ok = ok && left_robotDevice.view(left_encoders);
    ok = ok && left_robotDevice.view(left_axInfo);
    
    if (!ok) {
        yError("Problems acquiring interfaces\n");
        return 0;
    }

    Time::delay(3);

    /* __________________________________________ Gazebo read object pose ___________________________________________ */

    worldControl(0, "grasp-world", false);

    Property cfg;  
    cfg.put("device", "multipleanalogsensorsclient");  
    cfg.put("remote", "/cube-basestate");       // must match NWS "name" param  
    cfg.put("local",  "/object/cube-basestate");  
  
    PolyDriver object_client(cfg);  
    if (!object_client.isValid()) {
        yError("(object) Device not available.\n");
        return 0;
    }
  
    IPositionSensors* gz_object_pos = nullptr;  
    IOrientationSensors* gz_object_ori = nullptr;  
    object_client.view(gz_object_pos);  
    object_client.view(gz_object_ori);  
  
    /*
    yarp::sig::Vector p, rpy;  
    // Eigen::Map<Eigen::VectorXd> p_vec(p.data(), 3);
    double ts;  
  
    // object_ori->getOrientationSensorMeasureAsRollPitchYaw(0, rpy, ts);

    for (int i = 0; i < 100; i++)
    {
        if (gz_object_pos->getPositionSensorMeasure(0, p, ts))
        {
            Eigen::Map<Eigen::VectorXd> p_vec(p.data(), p.size());
            std::cout << "object pos:" << p_vec.transpose() << "\n\n";
            std::cout << "object pos string:" << p.toString() << "\n\n";
        }  
        Time::delay(0.05);  
    }
    */

    /* __________________________________________ Gazebo read icub pose ___________________________________________ */

    worldControl(0, "grasp-world", false);

    Property icub_cfg;  
    icub_cfg.put("device", "multipleanalogsensorsclient");  
    icub_cfg.put("remote", "/icub-basestate");       // must match NWS "name" param  
    icub_cfg.put("local",  "/robot/icub-basestate");  
  
    PolyDriver icub_client(icub_cfg);  
    if (!icub_client.isValid()) {
        yError("(icub) Device not available.\n");
        return 0;
    }
  
    IPositionSensors* icub_pos = nullptr;
    IOrientationSensors* icub_ori = nullptr;  
    icub_client.view(icub_pos);  
    icub_client.view(icub_ori);  
  
    /*
    yarp::sig::Vector p, rpy;  
    // Eigen::Map<Eigen::VectorXd> p_vec(p.data(), 3);
    double ts;  
  
    // icub_ori->getOrientationSensorMeasureAsRollPitchYaw(0, rpy, ts);

    for (int i = 0; i < 100; i++)
    {
        if (icub_pos->getPositionSensorMeasure(0, p, ts) && icub_ori->getOrientationSensorMeasureAsRollPitchYaw(0, rpy, ts))
        {
            Eigen::Map<Eigen::VectorXd> p_vec(p.data(), p.size());
            Eigen::Map<Eigen::VectorXd> rpy_vec(rpy.data(), rpy.size());
            std::cout << "icub pos:" << p_vec.transpose() << "\n\n";
            std::cout << "icub rpy:" << rpy_vec.transpose() << "\n\n";
        }  
        Time::delay(0.05);  
    }
    return 0;
    */

    /* __________________________________________ Force Sensor Right __________________________________________  */

    /* 
    Property r_ft_options;

    r_ft_options.put("device", "multipleanalogsensorsclient");
    r_ft_options.put("remote", "/icubSim/right_arm/FT");
    r_ft_options.put("local", "/right_ft_sensor");

    PolyDriver r_ft_sens(r_ft_options);

    if (!r_ft_sens.isValid()) {
        yError("FT Sensor (right) not available.\n");
        return 0;
    }

    ISixAxisForceTorqueSensors* r_ft;
    ok = r_ft_sens.view(r_ft);

    if (!ok) {
        yError("Problems acquiring (right) interfaces\n");
        return 0;
    }

    size_t num_r_ft = r_ft->getNrOfSixAxisForceTorqueSensors();
    std::cout << "Num sensors (right): " << num_r_ft << "\n";
    for (size_t i = 0; i < num_r_ft; i++) {
        std::string name;
        r_ft->getSixAxisForceTorqueSensorName(i, name);

        std::cout << "Name (right):" << name << "\n";
        const size_t ft_dim = 6;
        yarp::sig::Vector wrench(ft_dim);
        Eigen::Map<Eigen::VectorXd> wrench_vec(wrench.data(), ft_dim);
        double timestamp;
        r_ft->getSixAxisForceTorqueSensorMeasure(i, wrench, timestamp);
        std::cout << wrench_vec.transpose() << std::endl;
    }
    */

    /* __________________________________________ Force Sensor Left __________________________________________  */

    /*
    Property l_ft_options;

    l_ft_options.put("device", "multipleanalogsensorsclient");
    l_ft_options.put("remote", "/icubSim/left_arm/FT");
    l_ft_options.put("local", "/left_ft_sensor");

    PolyDriver l_ft_sens(l_ft_options);

    if (!l_ft_sens.isValid()) {
        yError("FT Sensor (left) not available.\n");
        return 0;
    }

    ISixAxisForceTorqueSensors* l_ft;
    ok = l_ft_sens.view(l_ft);

    if (!ok) {
        yError("Problems acquiring (left) interfaces\n");
        return 0;
    }

    size_t num_l_ft = l_ft->getNrOfSixAxisForceTorqueSensors();
    std::cout << "Num sensors (left): " << num_l_ft << "\n";
    for (size_t i = 0; i < num_l_ft; i++) {
        std::string name;
        l_ft->getSixAxisForceTorqueSensorName(i, name);

        std::cout << "Name (left):" << name << "\n";
        const size_t ft_dim = 6;
        yarp::sig::Vector wrench(ft_dim);
        Eigen::Map<Eigen::VectorXd> wrench_vec(wrench.data(), ft_dim);
        double timestamp;
        l_ft->getSixAxisForceTorqueSensorMeasure(i, wrench, timestamp);
        std::cout << wrench_vec.transpose() << std::endl;
    }
    */

    /* __________________________________________ ARM CONTROL START __________________________________________  */

    const std::vector<std::string> joint_list = {"shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_prosup", "wrist_pitch", "wrist_yaw"};
    const std::vector<std::string> ids = {"right_icub_arm", "left_icub_arm"};
    const std::vector<std::string> end_effector_frame_names = {"r_hand", "l_hand"};
    const std::string urdf_path = static_cast<std::string>(std::filesystem::current_path()) + "/../conf/model.urdf";

    std::vector<std::string> right_arm_joint_list;
    std::vector<std::string> left_arm_joint_list;

    for(auto it = joint_list.begin();
            it != joint_list.end();
            ++it)
    {
        right_arm_joint_list.push_back("r_" + *it);
        left_arm_joint_list.push_back("l_" + *it);
    }

    std::vector<std::vector<std::string>> list_of_joints;
    list_of_joints.push_back(right_arm_joint_list);
    list_of_joints.push_back(left_arm_joint_list);

    /* nmpc class init */
    dibiman::centralizedNMPC manip_controller(
        urdf_path,
        list_of_joints,
        end_effector_frame_names,
        ids);
  
    /* ______________ Exposing the created Models and Datas ______________ */

    pinocchio::Model & right_model = manip_controller.armList[0].model;
    pinocchio::Data & right_data = manip_controller.armList[0].data;
    pinocchio::FrameIndex & right_hand_id = manip_controller.armList[0].hand_id;

    pinocchio::Model & left_model = manip_controller.armList[1].model;
    pinocchio::Data & left_data = manip_controller.armList[1].data;
    pinocchio::FrameIndex & left_hand_id = manip_controller.armList[1].hand_id;

    /* ______________ Initial velocity and position vectors ______________ */

    /* Right Arm */
    int right_num_jnts;
    right_encoders->getAxes(&right_num_jnts);

    double* q_r_arr = new double[right_num_jnts]; /* joint positions */
    Eigen::Map<Eigen::VectorXd> q_r_vec(q_r_arr, right_model.nv);
    q_r_vec.setZero();

    double* v_r_arr = new double[right_num_jnts]; /* joint velocities */
    Eigen::Map<Eigen::VectorXd> v_r_vec(v_r_arr, right_model.nv);
    v_r_vec.setZero();

    right_encoders->getEncoders(q_r_arr);
    q_r_vec *= (M_PI/180);

    pinocchio::forwardKinematics(right_model, right_data, q_r_vec);
    pinocchio::updateFramePlacements(right_model, right_data);

    /* Left Arm */
    int left_num_jnts;
    left_encoders->getAxes(&left_num_jnts);

    double* q_l_arr = new double[left_num_jnts]; /* joint positions */
    Eigen::Map<Eigen::VectorXd> q_l_vec(q_l_arr, left_model.nv);
    q_l_vec.setZero();

    double* v_l_arr = new double[left_num_jnts]; /* joint velocities */
    Eigen::Map<Eigen::VectorXd> v_l_vec(v_l_arr, left_model.nv);
    v_l_vec.setZero();

    left_encoders->getEncoders(q_l_arr);
    q_l_vec *= (M_PI/180);

    pinocchio::forwardKinematics(left_model, left_data, q_l_vec);
    pinocchio::updateFramePlacements(left_model, left_data);

    /* ______________ Setting Up reference Frames ______________ */

    const Eigen::VectorXd right_arm_pos = right_data.oMf[right_hand_id].translation();
    const Eigen::VectorXd left_arm_pos = left_data.oMf[left_hand_id].translation();

    Eigen::VectorXd object_pos = .5f * (right_arm_pos + left_arm_pos);
    pinocchio::SE3 object_frame(Eigen::Matrix3d::Identity(), object_pos);

    std::map<std::string, Eigen::VectorXd> jointData;
    jointData[ids[0]] = q_r_vec;
    jointData[ids[1]] = q_l_vec;
    manip_controller.setObjectFrame(object_frame, jointData);

    /* ______________ Object Based Reference Creation (UNUSED FOR NOW) ______________ */

    const Eigen::VectorXd right_pos_ref = right_arm_pos + Eigen::Vector3d(0.00, 0.00, 0.15);
    const Eigen::VectorXd left_pos_ref = left_arm_pos + Eigen::Vector3d(0.00, 0.00, 0.15);

    const Eigen::Matrix3d r_R = right_data.oMf[right_hand_id].rotation();
    const pinocchio::SE3 r_H_ref(r_R, right_pos_ref);
    std::cout << "Right H ref : " << r_H_ref << std::endl;

    const Eigen::Matrix3d l_R = left_data.oMf[left_hand_id].rotation();
    const pinocchio::SE3 l_H_ref(l_R, left_pos_ref);
    std::cout << "Left H ref : " << l_H_ref << std::endl;

    /* ______________ Make the hands meet in the middle (basic grasp) ______________ */

    /*
    Eigen::VectorXd object_pos_ref = right_data.oMf[right_object_id].translation()
        + Eigen::Vector3d(0.02, 0.03, 0.06);

    Eigen::Vector3d rpy(0.0, M_PI/4.0, 0.0);
    Eigen::Matrix3d R_delta = pinocchio::rpy::rpyToMatrix(rpy);

    Eigen::Matrix3d object_Rot_ref = right_data.oMf[right_object_id].rotation() * R_delta;

    pinocchio::SE3 object_transform_ref(object_Rot_ref, object_pos_ref);
    */

    /* ______________ Inverse Kinematics ______________ */

    const int n_dim = right_model.nv;

    // const pinocchio::FrameIndex right_object_id = manip_controller.armList[0].object_id;
    Eigen::VectorXd q_r_ref_vec = dibiman::inverseKinematics(
            right_model,
            right_data,
            q_r_vec,
            r_H_ref,
            right_hand_id,
            "full"
            );
    std::cout << "q r ref :" << q_r_ref_vec.transpose() << std::endl;
    Eigen::VectorXd v_r_ref_vec(n_dim);
    v_r_ref_vec.setZero();

    Eigen::VectorXd u_r_ref_vec = pinocchio::rnea(right_model, right_data, q_r_ref_vec, v_r_ref_vec, v_r_ref_vec);
    std::cout << "u r ref: " << u_r_ref_vec.transpose() << std::endl;

    // const pinocchio::FrameIndex left_object_id = manip_controller.armList[1].object_id;
    Eigen::VectorXd q_l_ref_vec = dibiman::inverseKinematics(
            left_model,
            left_data,
            q_l_vec,
            l_H_ref,
            left_hand_id,
            "full"
            );

    std::cout << "q l ref :" << q_l_ref_vec.transpose() << std::endl;
    Eigen::VectorXd v_l_ref_vec(n_dim);
    v_l_ref_vec.setZero();

    Eigen::VectorXd u_l_ref_vec = pinocchio::rnea(left_model, left_data, q_l_ref_vec, v_l_ref_vec, v_l_ref_vec);
    std::cout << "u l ref: " << u_l_ref_vec.transpose() << std::endl;

    /* ____________ Solver Structs Init _________________ */

    std::map<std::string, dibiman::armTargetParams> bimanual_solve_params;

    dibiman::armTargetParams right_params(q_r_vec.size());
    dibiman::armTargetParams left_params(q_l_vec.size());

    bimanual_solve_params[ids[0]] = right_params;
    bimanual_solve_params[ids[1]] = left_params;

    right_params.x0 << q_r_vec, v_r_vec;
    right_params.xref << q_r_ref_vec, v_r_ref_vec;
    right_params.uref = u_r_ref_vec;

    left_params.x0 << q_l_vec, v_l_vec;
    left_params.xref << q_l_ref_vec, v_l_ref_vec;
    left_params.uref = u_l_ref_vec;

    std::map<std::string, casadi::DM> u_star;
    u_star[ids[0]] = casadi::DM::zeros(n_dim);
    u_star[ids[1]] = casadi::DM::zeros(n_dim);

    int joint_idx_arr[] = {0, 1, 2, 3, 4, 5, 6};
    
    for (int i = 0; i < static_cast<int>(n_dim); i++) {
        right_controlMode->setControlMode(i, VOCAB_CM_TORQUE);
        left_controlMode->setControlMode(i, VOCAB_CM_TORQUE);
    }

    worldControl(0, "grasp-world", false);

    for (;;) {

        left_encoders->getEncoders(q_l_arr);
        right_encoders->getEncoders(q_r_arr);

        left_encoders->getEncoderSpeeds(v_l_arr);
        right_encoders->getEncoderSpeeds(v_r_arr);

        left_params.x0 << q_l_vec, v_l_vec;
        left_params.x0 *= (M_PI/180);

        right_params.x0 << q_r_vec, v_r_vec;
        right_params.x0 *= (M_PI/180);

        bimanual_solve_params[ids[0]] = right_params;
        bimanual_solve_params[ids[1]] = left_params;

        u_star = manip_controller.solve(bimanual_solve_params);
        double* u_r = u_star.at(ids[0]).ptr();
        double* u_l = u_star.at(ids[1]).ptr();

        bool ok = right_torqueControl->setRefTorques(static_cast<int>(n_dim), joint_idx_arr, u_r);
        if (!ok) return 1;
        ok = left_torqueControl->setRefTorques(static_cast<int>(n_dim), joint_idx_arr, u_l);
        if (!ok) return 1;

        // static double timestamp;
        // static const size_t ft_dim = 6;

        // static yarp::sig::Vector r_wrench(ft_dim);
        // static Eigen::Map<Eigen::VectorXd> r_wrench_vec(r_wrench.data(), ft_dim);

        // static yarp::sig::Vector l_wrench(ft_dim);
        // static Eigen::Map<Eigen::VectorXd> l_wrench_vec(l_wrench.data(), ft_dim);

        // r_ft->getSixAxisForceTorqueSensorMeasure(0, r_wrench, timestamp);
        // l_ft->getSixAxisForceTorqueSensorMeasure(0, l_wrench, timestamp);

        // std::cout << "Right x: " << right_params.x0.transpose() << "\n";
        // std::cout << "Right Wrench: " << r_wrench_vec.transpose() << "\n";
        // std::cout << "Right Torque: " << u_star.at(ids[0]) << "\n\n";

        // std::cout << "Left x: " << left_params.x0.transpose() << "\n";
        // std::cout << "Left Wrench: " << l_wrench_vec.transpose() << "\n";
        // std::cout << "Left Torque: " << u_star.at(ids[1]) << "\n\n";

        Time::delay(0.03);

        worldControl(5, "grasp-world");

        // pinocchio::forwardKinematics(reduced_model, data, q_sens_Vec);  
        // pinocchio::updateFramePlacements(reduced_model, data);  

        /* send info to other node(arm) */
    }

    right_robotDevice.close();
    left_robotDevice.close();
    
    return 0;
}
