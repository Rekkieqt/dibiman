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
#include <yarp/dev/IAxisInfo.h>
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
#include "pinocchio/algorithm/frames.hpp"
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
  const int joint_id = 6; /* one extra because of universe, or just count from 1 */
  const double eps = 1e-4;
  const int IT_MAX = 800;
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
    bool single_mode = rf.find("single").asBool();
    
    if (robotName=="")
    {
        cout << "Failed loading config file!" << endl;
        return -1;
    }

    /* ++++++ YARP NETWORKING ++++++ */

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
    IEncoders *armSensors;
    IAxisInfo *axInfo;

    /* check if interfaces are available */
    bool ok;
    ok = robotDevice.view(controlMode);
    ok = ok && robotDevice.view(torqueControl);
    ok = ok && robotDevice.view(positionControl);
    ok = ok && robotDevice.view(armSensors);
    ok = ok && robotDevice.view(axInfo);
    
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

    /* +++++++++++ Loading the Model ++++++++++++++*/
    Model model, reduced_model;
    pinocchio::urdf::buildModel(urdf_filename, model);

    /* arm joints */
    std::vector<std::string> arm_config = {"shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_prosup", "wrist_pitch"};
    /*, "_wrist_yaw"}; unused joint */

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

    /* sample neutral config */
    Eigen::VectorXd q_full = pinocchio::neutral(model);

    /* build the reduced model */
    reduced_model = pinocchio::buildReducedModel(model, keep_locked_by_id, q_full);

    /* Create data required by the algorithms */
    Data data(reduced_model);

    /* +++++++++++ ARM CONTROL START ++++++++++++++*/
    /* remote controller */
    int idx_joints[] = {0, 1, 2, 3, 4, 5};
    int all_arm_joints = 0;
    armSensors->getAxes(&all_arm_joints);

    /* Controlled joints */
    double* q_sens = new double[joints]; /* joint positions */
    double* v_sens = new double[joints]; /* joint velocities */
    double* a_sens = new double[joints]; /* joint accelerations */
    double* u_sens = new double[joints]; /* tau -> torque at the joints */ 

    /* dynamic link between double arrays and eigen arrays */
    Eigen::Map<Eigen::VectorXd> q_sens_Vec(q_sens, joints);
    // Eigen::Map<Eigen::VectorXd> qd_meas_v(qd_all, joints);
    // Eigen::Map<Eigen::VectorXd> qdd_meas_v(qdd_meas, joints);
    // Eigen::Map<Eigen::VectorXd> tau_v(tau, joints);
    // Eigen::Map<Eigen::VectorXd> tau_meas_v(tau_meas, joints);

    std::string logfile = "../../logs/" + partName + ".csv";
    fstream file(logfile, ios::out | ios::trunc);
    logHeader(file, joints);

    /* set control mode for arm */
    int modes[] = {VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE, VOCAB_CM_TORQUE};
    // controlMode->setControlModes(joints, idx_joints, modes);

    /* solve inverse kinematics */
    /* iKin(reduced_model, data, q_ref_v, q_meas_v); */
    /* log the ref angles */
    // std::string reffile = "../../logs/" + partName + "_ref.csv";
    // fstream ref(reffile, ios::out | ios::trunc);
    /* logRef(ref, q_ref_v, joints); */
    // ref.close();
    int handID = reduced_model.getFrameId(arm_prefix + "hand");

    for (int i = 0; i < 2; i++) {

        for (int j = 0; j < std::size(idx_joints) ; j++) {
          armSensors->getEncoder(idx_joints[j], &q_sens[j]);
          q_sens[j] = (M_PI/180) * q_sens[j];
          std::cout << q_sens[j] << std::endl;
        }
        pinocchio::forwardKinematics(reduced_model, data, q_sens_Vec);  
        pinocchio::updateFramePlacements(reduced_model, data);  
        std::cout << data.oMf[handID] << std::endl;

        /* send torque commands */
        // ok = torqueControl->setRefTorques(joints, idx_joints, tau);

        /* write to file */
        // logData(file, tau_v, q_meas_v, joints);

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
        
        yarp::os::Time::delay(0.05);
    }
    /* cleanup, effectively useless because at the moment I ctrl+c from while */
    /* later can put this into 'graceful' exit with interrupt ... */

    robotDevice.close();
    file.close();
    
    return 0;
}
