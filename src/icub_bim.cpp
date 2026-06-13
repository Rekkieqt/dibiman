/* yarp libraries */
#include "yarp/os/Bottle.h"
#include "yarp/os/BufferedPort.h"
#include "yarp/os/Log.h"
#include <vector>
#include <yarp/os/Network.h>
#include <yarp/os/Time.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/ITorqueControl.h>
#include <yarp/dev/IControlMode.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>
#include <yarp/math/Math.h>

/* cpp libraries */
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <list>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

/* Pinocchio libraries */
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

/* NMPC libraries */
// #include "../include/nmpc_fmpc/FmpcSolver.h"

using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::math;
using namespace std;
using namespace pinocchio;

/* NMPC library testing (START) */

/* NMPC library testing (END) */

template<typename T>
bool is_in_vector(const std::vector<T> & vector, const T & elt) {
  return vector.end() != std::find(vector.begin(), vector.end(), elt);
}

bool iKin(pinocchio::Model & model, pinocchio::Data & data, Eigen::VectorXd & q, Eigen::Ref<Eigen::VectorXd> q_m) {

  /* constants */
  const int joint_id = 7; /* one extra because of universe, or just count from 1 */
  const double eps = 1e-4;
  const int IT_MAX = 500;
  const double DT = 1e-1;
  const double damp = 1e-6;
  bool success = false;

  /* Eigen::VectorXd q = pinocchio::neutral(model); */
  // int q_size = 7;
  // Eigen::Map<Eigen::VectorXd> q(q_meas, q_size);
  q = q_m;
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::SE3 oMdes(data.oMi[joint_id].rotation(), data.oMi[joint_id].translation() + Eigen::Vector3d(0.0, 0.0, 0.15));
  cout << "quaternion curr" << oMdes.rotation() << endl;

  pinocchio::Data::Matrix6x J(6, model.nv);
  J.setZero();

  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d err;
  Eigen::VectorXd v(model.nv);
  for (int i = 0;; i++) {
    const pinocchio::SE3 iMd = data.oMi[joint_id].actInv(oMdes);
    err = pinocchio::log6(iMd).toVector(); /* joint frame */

    if (err.norm() < eps) { /* stoping criteria = error */
      success = true;
      break;
    }
    if (i >= IT_MAX) { /* stopping criteira = iterations */
      success = false;
      break;
    }
    pinocchio::computeJointJacobian(model, data, q, joint_id, J); /* Jacobian joint frame */
    pinocchio::Data::Matrix6 Jlog;
    pinocchio::Jlog6(iMd.inverse(), Jlog);
    J = -Jlog * J;
    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += damp;
    v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
    q = pinocchio::integrate(model, q, v * DT);
    
  }
  cout << "quaternion ref" << data.oMi[joint_id].rotation() << endl;
  std::cout << "\nq_ref: " << q.transpose() << std::endl;
  return success;
}

/* logging function */

void logData(std::fstream & logfile, Eigen::Ref<Eigen::VectorXd> tau, Eigen::Ref<Eigen::VectorXd> qread, int size) {
  /* log torques */
  for (int i=0; i < size; ++i) {
    logfile << qread[i] << ",";
  }
  /* log torques */
  for (int i=0; i < size; ++i) {
    logfile << tau[i] << ",";
  }
  logfile << std::endl;
}

void logHeader(std::fstream & logfile, int size) {
  /* log readings */
  for (int i=0; i < size; ++i) {
    logfile << "joint_meas" << i << ",";
  }
  /* log torques */
  for (int i=0; i < size; ++i) {
    logfile << "torque " << i << ",";
  }
  logfile << std::endl;
}

void logRef(std::fstream & reffile, Eigen::VectorXd& qref, int size) {
  /* log refs */
  for (int i=0; i < size; ++i) {
    reffile << "joint_ref" << i << ",";
  }
  reffile << std::endl;
  /* log torques */
  for (int i=0; i < size; ++i) {
    reffile << qref[i] << ",";
  }
  reffile << std::endl;
}


int main(int argc, char **argv)
{
    /* where to put example nmpc */

    /* dynamic parameter loading with rf*/
    ResourceFinder rf;
    rf.configure(argc, argv);

    /* findGroup -> find a list and save it as a Bottle, useful for later */
    std::string robotName = rf.find("robot").asString();
    std::string partName = rf.find("part").asString();
    std::string local = rf.find("local").asString();
    std::string remote = rf.find("remote").asString();
    std::string comm = rf.find("comm").asString();
    std::string arm_prefix = rf.find("prfx").asString();
    std::string urdf_filename = rf.find("model").asString(); 

    int joints = rf.find("joints").asInt32();
    double theta_dd_ref = rf.find("theta_dd").asFloat64();
    double theta_d_ref = rf.find("theta_d").asFloat64();
    double kv = rf.find("kv").asFloat64();
    double kp = rf.find("kp").asFloat64();
    bool single_mode = rf.find("single").asBool();
    
    if (robotName=="")
    {
        cout << "Failed loading config file!" << endl;
        return -1;
    }

    /* loading urdf with pinocchio*/
    Model model, reduced_model;
    pinocchio::urdf::buildModel(urdf_filename, model);

    /* arm joints */
    std::vector<std::string> arm_config = {"_shoulder_pitch", "_shoulder_roll", "_shoulder_yaw", "_elbow", "_wrist_prosup", "_wrist_pitch", "_wrist_yaw"};

    /* joints to use */
    for (auto it = arm_config.begin(); it != arm_config.end(); ++it){
      *it = arm_prefix + *it;
    }
    std::vector<JointIndex> keep_unlocked_by_id, keep_locked_by_id;
    for (std::vector<std::string>::const_iterator it = arm_config.begin();
        it != arm_config.end();
        ++it){
      const std::string & joint_name = *it;
      if (model.existJointName(joint_name)){
        keep_unlocked_by_id.push_back(model.getJointId(joint_name));
      }
    }

    /* invert the list */
    for (JointIndex joint_id = 1; joint_id < model.joints.size(); ++joint_id) {
      const std::string joint_name = model.names[joint_id];
      if (is_in_vector(arm_config, joint_name)){
        continue;
      }
      else {
        keep_locked_by_id.push_back(joint_id);
      }
    }

    /* sample random config */
    Eigen::VectorXd q_full = randomConfiguration(model);

    /* build the reduced model */
    reduced_model = pinocchio::buildReducedModel(model, keep_locked_by_id, q_full);

    /* Create data required by the algorithms */
    Data data(reduced_model);

    /* print random config */
    /* pinocchio example end */

    /* configuring in and out ports */
    /* where to connect */
    std::string remotePorts="/";
    remotePorts+=robotName;
    remotePorts+="/";
    remotePorts+=partName;

    /* name of local port */
    std::string localPorts= rf.find("local").asString();

    /* connect to simulated control board*/
    Property options;
    options.put("device", "remote_controlboard");
    options.put("local", localPorts.c_str());   
    options.put("remote", remotePorts.c_str()); 

    /* create a device */
    PolyDriver robotDevice(options);
    if (!robotDevice.isValid()) {
        yError("Device not available.\n");
        return 0;
    }

    /* create interfaces */
    IControlMode *controlMode;
    ITorqueControl *torqueControl;
    IPositionControl *positionControl;
    IEncoders *q_sens;

    /* check if interfaces are available */
    bool ok;
    ok = robotDevice.view(controlMode);
    ok = ok && robotDevice.view(torqueControl);
    ok = ok && robotDevice.view(positionControl);
    ok = ok && robotDevice.view(q_sens);
    
    if (!ok) {
        yError("Problems acquiring interfaces\n");
        return 0;
    }

    /* establish comms between right and left arm */
    /* from where to write */
    std::string inPort = rf.find("inport").asString();
    /* where to receive */
    std::string outPort = rf.find("outport").asString();
    /* where to write */
    std::string writePort = rf.find("writeport").asString();

    BufferedPort<Bottle> recvPort;
    BufferedPort<Bottle> sendPort;

    recvPort.open(inPort);
    sendPort.open(outPort);

    /* yarp networking */
    Network yarp;

    /* wait for the other node to come online */
    while (!(yarp.connect(outPort, writePort) || single_mode)) {
        cout << "Waiting for the other node to connect ..." << endl;
        yarp::os::Time::delay(3);
    }

    /* remote controller */
    int arm_joints = 0;
    int idx_joints[] = {0, 1, 2, 3, 4, 5, 6};

    q_sens->getAxes(&arm_joints);

    double* q_all = new double[arm_joints]; /* encoder positions */
    double* qd_all = new double[arm_joints]; /* joint velocities */
    double* qdd_meas = new double[joints]; /* dq^2/dt^2 -> theta 2dot of the revolute joints */
    double* tau = new double[joints]; /* tau -> torque vector at the joints */ 
    double* tau_meas = new double[arm_joints]; /* tau_meas -> measured torque vector at the joints */ 

    /* dynamic link between double arrays and eigen arrays */
    Eigen::Map<Eigen::VectorXd> q_meas_v(q_all, joints);
    Eigen::Map<Eigen::VectorXd> qd_meas_v(qd_all, joints);
    Eigen::Map<Eigen::VectorXd> qdd_meas_v(qdd_meas, joints);
    Eigen::Map<Eigen::VectorXd> tau_v(tau, joints);
    Eigen::Map<Eigen::VectorXd> tau_meas_v(tau_meas, joints);
    Eigen::VectorXd q_ref_v;

    /* create target velocities and accelerations of q */
    Eigen::VectorXd qd_ref_v = theta_d_ref * Eigen::VectorXd::Ones(joints);
    Eigen::VectorXd qdd_ref_v = theta_dd_ref * Eigen::VectorXd::Ones(joints);
    Eigen::VectorXd error_vector(joints);
    Eigen::VectorXd speed_error(joints);
    Eigen::VectorXd pos_error(joints);
    error_vector.setZero();
    Eigen::VectorXd integral_vector(joints);
    integral_vector.setZero();

    /* diagonal matrices of gains */
    Eigen::VectorXd Kv = kv * Eigen::VectorXd::Ones(joints);
    Eigen::VectorXd Kp = kp * Eigen::VectorXd::Ones(joints);
    Eigen::VectorXd Ki = kv * Eigen::VectorXd::Ones(joints);
    
    std::string logfile = "../../logs/" + partName + ".csv";
    fstream file(logfile, ios::out | ios::trunc);
    logHeader(file, joints);

    /* set control mode for arm */
    int modes[] = {VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE};
    controlMode->setControlModes(joints, idx_joints, modes);

    /* get measurements */
    while(!q_sens->getEncoders(q_all));
    // q_ref_v = q_meas_v + 20*Eigen::VectorXd::Ones(joints);
    // cout << "qref:" << q_ref_v.transpose() << endl;

    /* solve inverse kinematics */
    iKin(reduced_model, data, q_ref_v, q_meas_v);

    /* log the ref angles */
    std::string reffile = "../../logs/" + partName + "_ref.csv";
    fstream ref(reffile, ios::out | ios::trunc);
    logRef(ref, q_ref_v, joints);
    ref.close();

    for(;;) {
        /* get measurements */
        while(!q_sens->getEncoders(q_all));
        while(!q_sens->getEncoderSpeeds(qd_all));

        /* simple linearizing control */
        speed_error = qd_ref_v - qd_meas_v;
        pos_error = q_ref_v - q_meas_v;
        // error_vector = qdd_ref_v + Kv.asDiagonal() * (speed_error) + Kp.asDiagonal() * (pos_error); /* error vector */
        error_vector = Kv.asDiagonal() * (speed_error) + Kp.asDiagonal() * (pos_error); /* error vector */
        pinocchio::rnea(reduced_model, data, q_meas_v, qd_meas_v.setZero(), qdd_meas_v.setZero()); /* obtain C matrix, located in data.tau*/
        pinocchio::crba(reduced_model, data, q_meas_v); /* obtain M(q) matrix, data.M */
        integral_vector += Kp.asDiagonal()*pos_error;
        tau_v = error_vector + integral_vector + data.tau;

        /* send torque commands */
        ok = torqueControl->setRefTorques(joints, idx_joints, tau);

        /* write to file */
        logData(file, tau_v, q_meas_v, joints);

        /* send info to other node(arm) */
        Bottle *b = recvPort.read(false);
        if (b!=NULL) {
            /* receive communication works */
            // cout << "Data received!\n";
            // cout << "got " << b->toString().c_str();
        }
        Bottle& s = sendPort.prepare();
        s.clear();
        s.addString(inPort);
        sendPort.write();
        
        if (pos_error.norm() < 0.01) { /* arbitrary error criteria */
          cout << "Desired acc achieved!!" << endl;
          break;
        }
        yarp::os::Time::delay(0.01);
    }
    /* cleanup, effectively useless because at the moment I ctrl+c from while */
    /* later can put this into 'graceful' exit with interrupt ... */

    robotDevice.close();
    file.close();
    
    return 0;

