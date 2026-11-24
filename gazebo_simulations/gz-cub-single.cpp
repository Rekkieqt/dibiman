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

/* OpenMP */
#include "omp.h"

/* NMPC libraries */
#include "dibiman/nmpc/distr-nmpc.hpp"
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

void 
worldControl(uint64_t _steps, const std::string & _world_name, bool _paused=true)
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

int 
main(int argc, char **argv)
{
    /* __________________________________________ YARP NETWORKING __________________________________________ */

    /* findGroup -> find a list and save it as a Bottle, useful for later */
    const std::string right_robotName = "icubSim"; /* icubSim */
    const std::string right_partName = "right_arm"; /* {ARM}_arm */
    const std::string right_local = "/right_arm_controller"; /* /{ARM}_arm_controller */

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

    /* __________________________________________ Gazebo read object pose ___________________________________________ */

    worldControl(0, "grasp-world", false);
    Time::delay(5);

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
  
    yarp::sig::Vector gz_obj_p;
    double ts;
    while(!gz_object_pos->getPositionSensorMeasure(0, gz_obj_p, ts)) Time::delay(0.05);
    Eigen::Map<Eigen::VectorXd> gz_obj_p_vec(gz_obj_p.data(), gz_obj_p.size());
    std::cout << "cube pose: " << gz_obj_p_vec.transpose() << std::endl;
  
    /* __________________________________________ Gazebo read icub pose ___________________________________________ */

    worldControl(0, "grasp-world", false);
    Time::delay(5);

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
  
    yarp::sig::Vector gz_icub_p;
    while (!icub_pos->getPositionSensorMeasure(0, gz_icub_p, ts)) Time::delay(0.05);
    Eigen::Map<Eigen::VectorXd> gz_icub_p_vec(gz_icub_p.data(), gz_icub_p.size());
    std::cout << "icub pose: " << gz_icub_p_vec.transpose() << std::endl;

    /* __________________________________________ Force Sensor Right __________________________________________  */

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

    /* __________________________________________ ARM CONTROL START __________________________________________  */

    const std::vector<std::string> joint_list = {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"};
    const std::vector<std::string> ids = {"right_icub_arm", "left_icub_arm"};
    const std::string ee_frame = "r_hand_dh_frame";
    const std::string model_path = "/home/ubuntu/dibiman/conf/model_FT.urdf";

    pinocchio::Model model = dibiman::returnReducedModel(model_path, joint_list);
    pinocchio::Data data = pinocchio::Data(model);

    const int horizon = 50;
    const int n_threads = 4;
    omp_set_num_threads(n_threads);
    dibiman::DecentralizedNMPC man(model, ee_frame, horizon, n_threads);
    man.set_dt(0.01);
  
    /* ______________ Initial velocity and position vectors ______________ */

    std::cerr << "Linking yarp q and v to Eigen\n\n";
    constexpr double to_pi = M_PI/180;

    /* Right Arm */
    int n_joints;
    right_encoders->getAxes(&n_joints);

    double* q_arr = new double[n_joints]; /* joint positions */
    Eigen::Map<Eigen::VectorXd> q(q_arr, model.nv);
    q.setZero();

    double* v_arr = new double[n_joints]; /* joint velocities */
    Eigen::Map<Eigen::VectorXd> v(v_arr, model.nv);
    v.setZero();

    double* tau_arr = new double[n_joints];
    Eigen::Map<Eigen::VectorXd> tau(tau_arr, model.nv);

    Eigen::VectorXd x(model.nv + model.nv);

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    /* ______________ Object grasp movement 1 - hover ______________ */

    Eigen::VectorXd cub_to_obj_vec = gz_obj_p_vec - gz_icub_p_vec;
    std::cout << "gazebo obj pose rel: " << cub_to_obj_vec.transpose() << std::endl;

    const Eigen::Vector3d pos_ref = cub_to_obj_vec + Eigen::Vector3d(0.0, 0.1, 0.0);

    const Eigen::Vector3d right_rpy(M_PI/2.0f, M_PI, 0);
    const Eigen::Matrix3d r_R = pinocchio::rpy::rpyToMatrix(right_rpy);
    pinocchio::SE3 r_H_ref(r_R, cub_to_obj_vec);
    std::cout << "Right H ref : " << r_H_ref << std::endl;

    /* ______________ Inverse Kinematics ______________ */

    Eigen::VectorXd q_ref = dibiman::inverseKinematics(
            model,
            data, 
            q, 
            r_H_ref,
            model.getFrameId(ee_frame));

    man.setRef(q_ref);

    /* ____________ Solver Structs Init _________________ */

    const int joint_idx_arr[] = {0, 1, 2, 3, 4, 5, 6};
    
    for (int i = 0; i < model.nv; i++) {
        right_controlMode->setControlMode(i, VOCAB_CM_TORQUE);
    }

    worldControl(0, "grasp-world", false);

    std::cout << "Initiating\n\n";
    
    Time::delay(5);

    for (int i = 0; ; i++) {

        right_encoders->getEncoders(q_arr);
        right_encoders->getEncoderSpeeds(v_arr);

        x << q, v;
        x *= to_pi;
        tau = man.solve(x);
        std::cout << "State: " << x.transpose() << "\n\n";
        std::cout << "Torque: " << tau.transpose() << "\n\n";

        bool ok = right_torqueControl->setRefTorques(model.nv, joint_idx_arr, tau_arr);
        if (!ok) return 1;

        static double timestamp;
        const size_t ft_dim = 6;

        Time::delay(0.05);

        pinocchio::forwardKinematics(model, data, q * to_pi);
        pinocchio::updateFramePlacements(model, data);

        /*
        pinocchio::Motion err_r = pinocchio::log6(r_H_ref.actInv(right_hand_frame));
        double err_r_norm = err_r.toVector().norm();

        if (i%10 == 0) {
            std::cout << "right error: " << err_r_norm << "\n\n";

            yarp::sig::Vector r_wrench(ft_dim);
            Eigen::Map<Eigen::VectorXd> r_wrench_vec(r_wrench.data(), ft_dim);

            r_ft->getSixAxisForceTorqueSensorMeasure(0, r_wrench, timestamp);

            std::cout << "Right Wrench: " << r_wrench_vec.transpose() << "\n";
        }

        bool err_test = (err_r_norm < eps);
        if (err_test) {
            std::cout << "Initiating Closure" << std::endl;
            right_params.xref << q_r_ref_vec_2, v_r_ref_vec;
            right_params.uref = u_r_ref_vec_2;

        }
        */

        worldControl(5, "grasp-world");

    }

    right_robotDevice.close();
    
    return 0;
}
