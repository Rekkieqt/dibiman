#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <numeric>
#include <cstdlib>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>
#include <iCub/ctrl/math.h>
#include <iCub/ctrl/adaptWinPolyEstimator.h>
#include <iCub/iDyn/iDyn.h>
#include <iCub/iDyn/iDynBody.h>
#include <iCub/iKin/iKinFwd.h>
#include <iCub/skinDynLib/skinContact.h>

#include <iDynTree/YARPConversions.h>
#include <iDynTree/Utils.h>
#include <iDynTree/Wrench.h>
#include <iDynTree/MatrixFixSize.h>
#include <iDynTree/EigenHelpers.h>
#include <iDynTree/KinDynComputations.h>
#include <iDynTree/ExtWrenchesAndJointTorquesEstimator.h>
#include <iDynTree/skinDynLibConversions.h>
#include <iDynTree/ContactStateMachine.h>
// #include <iDynTree/Estimation/KalmanFilter.h>
#include <iDynTree/Sensors.h>
#include <iDynTree/ModelLoader.h>
#include <iDynTree/KinDynComputations.h>

#include "forceConv.h"

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;
using namespace yarp::dev;
using namespace iCub::ctrl;
using namespace iCub::iDyn;
using namespace iCub::iKin;
using namespace iCub::skinDynLib;
using namespace std;

void baseConversion::setStiffMode()
{
     if (iint_arm_left)
     {
         for (int i=0; i<7; i++)
            iint_arm_left->setInteractionMode(i,VOCAB_IM_STIFF);
     }
     if (iint_arm_right)
     {
         for (int i=0; i<7; i++)
            iint_arm_right->setInteractionMode(i,VOCAB_IM_STIFF);
     }
     if (iint_torso)
     {
         for (int i=0; i<3; i++)
            iint_torso->setInteractionMode(i,VOCAB_IM_STIFF);
     }

     if (icmd_arm_left)
     {
         for (int i=0; i<7; i++)
             {
                 int mode =0; icmd_arm_left->getControlMode(i, &mode);
                 if (mode==VOCAB_CM_TORQUE)  icmd_arm_left->setControlMode(i, VOCAB_CM_POSITION);
             }
     }

     if (icmd_arm_right)
     {
         for (int i=0; i<7; i++)
             {
                 int mode =0; icmd_arm_right->getControlMode(i, &mode);
                 if (mode==VOCAB_CM_TORQUE)  icmd_arm_right->setControlMode(i, VOCAB_CM_POSITION);
             }
     }

     if (icmd_torso)
     {
         for (int i=0; i<3; i++)
             {
                 int mode =0; icmd_torso->getControlMode(i, &mode);
                 if (mode==VOCAB_CM_TORQUE)  icmd_torso->setControlMode(i, VOCAB_CM_POSITION);
             }
     }
}

baseConversion::baseConversion(int _rate, int _FT, int _skin, string _arm, string _robot_name, string _local_name,
                                bool _autoconnect,ISixAxisForceTorqueSensors* m_left_arm_FT, ISixAxisForceTorqueSensors* m_right_arm_FT,
                                float _gain_index, float _gain_middle, float _gain_ring, float _gain_little, float _gain_thumb, float _gain_palm):
                                 PeriodicThread((double)_rate/1000.0), robot_name(_robot_name), local_name(_local_name),
                                 zero_sens_tolerance (1e-12),m_left_arm_FT(m_left_arm_FT), m_right_arm_FT(m_right_arm_FT), gain_index(_gain_index),
                                 gain_middle(_gain_middle), gain_ring(_gain_ring), gain_little(_gain_little), gain_thumb(_gain_thumb), gain_palm(_gain_palm)
{
    status_queue_size = 10;
    autoconnect = _autoconnect;
    FT = _FT;
    skin = _skin;
    arm = _arm;

    arm_left_bwd_1 = new iCubArmDyn("left"); // ,KINBWD_WREBWD
    arm_right_bwd_1 = new iCubArmDyn("right");

    arm_left_bwd_1->releaseLink(0);
    arm_left_bwd_1->releaseLink(1);
    arm_left_bwd_1->releaseLink(2);

    arm_left_bwd = new iCubArmDyn("left",KINBWD_WREBWD);
    arm_right_bwd = new iCubArmDyn("right",KINBWD_WREBWD);

    icub = new iCubWholeBody (icub_type,DYNAMIC, VERBOSE);

    //icub      = new iCubWholeBody(icub_type, DYNAMIC, VERBOSE);
    //icub_sens = new iCubWholeBody(icub_type, DYNAMIC, VERBOSE);
    left_FT = new iDynSensorArm(arm_left_bwd, DYNAMIC, VERBOSE); // icub_type
    right_FT = new iDynSensorArm(arm_right_bwd, DYNAMIC, VERBOSE);
    left_skin = new OneChainNewtonEuler(arm_left_bwd,"left",DYNAMIC,VERBOSE);
    right_skin = new OneChainNewtonEuler(arm_right_bwd,"right",DYNAMIC,VERBOSE);

    left_gravity = new iDynInvSensorArm(arm_left_bwd_1,"left",DYNAMIC,VERBOSE);
    right_gravity = new iDynInvSensorArm(arm_right_bwd_1,"right",DYNAMIC,VERBOSE);

    first = true;

    //std::string modelAbsolutePath = yarp::os::ResourceFinder::getResourceFinderSingleton().findFileByName("model_2_5_visuomanip.urdf");
    bool ok = estimator.loadModelAndSensorsFromFile("/home/vislab/thesis_diogo_silva/Code_Implementation/forceConversion/iCubGazeboV2_5_visuomanip/model_2_5_visuomanip.urdf");

    iDynTree::ModelLoader modelLoader;
    bool ok_2 = modelLoader.loadModelFromFile("/home/vislab/thesis_diogo_silva/Code_Implementation/forceConversion/iCubGazeboV2_5_visuomanip/model_2_5_visuomanip.urdf");
    iDynTree::KinDynComputations kinDynComp;
    kinDynComp.loadRobotModel(modelLoader.model());
    
    //--------------INTERFACE INITIALIZATION-------------//

    port_FT_RA = new BufferedPort<Vector>;
    port_FT_LA = new BufferedPort<Vector>;

    port_FT_RA_base = new BufferedPort<Vector>;
    port_FT_LA_base = new BufferedPort<Vector>;
    port_skin_events = new BufferedPort<skinContactList>;

    port_wrench_l_skin = new BufferedPort<Vector>;
    port_wrench_r_skin = new BufferedPort<Vector>;

    port_l_gravity = new BufferedPort<Vector>;
    port_r_gravity = new BufferedPort<Vector>;

    port_joints_l_arm = new BufferedPort<Vector>;
    port_joints_r_arm = new BufferedPort<Vector>;

    port_inertials = new BufferedPort<Vector>;

    port_FT_RA->open(string("/"+local_name+"/right_arm_analog:i").c_str());
    port_FT_LA->open(string("/"+local_name+"/left_arm_analog:i").c_str());
    port_FT_RA_base->open(string("/"+local_name+"/right_base_analog:o").c_str());
    port_FT_LA_base->open(string("/"+local_name+"/left_base_analog:o").c_str());
    port_skin_events->open(string("/"+local_name+"/skin_events_conv:i").c_str());
    port_wrench_l_skin->open(string("/"+local_name+"/wrench_skin_l_arm:o").c_str());
    port_wrench_r_skin->open(string("/"+local_name+"/wrench_skin_r_arm:o").c_str());

    port_l_gravity->open(string("/"+local_name+"/wrench_gravity_left:o").c_str());
    port_r_gravity->open(string("/"+local_name+"/wrench_gravity_right:o").c_str());

    port_joints_l_arm->open(string("/"+local_name+"/arm_l_joints:i").c_str());
    port_joints_r_arm->open(string("/"+local_name+"/arm_r_joints:i").c_str());
    port_inertials->open(string("/"+local_name+"/inertials:i").c_str());

    yInfo ("Waiting for port connections");
    if (autoconnect)
    {
        //Network::connect(string("/wholeBodyDynamics/filteredFT/l_arm_ft").c_str(),string("/"+local_name+"/left_arm_analog:i").c_str(),"tcp",false); // /wholeBodyDynamics/filteredFT/l_arm_ft
        Network::connect(string("/wholeBodyDynamics/filteredFT/r_arm_ft").c_str(),string("/"+local_name+"/right_arm_analog:i").c_str(),"tcp",false); // /wholeBodyDynamics/filteredFT/r_arm_ft
        //Network::connect(string("/icubSim/left_arm/state:o").c_str(),string("/"+local_name+"/arm_l_joints:i").c_str(),"tcp",false);
        Network::connect(string("/icubSim/right_arm/state:o").c_str(),string("/"+local_name+"/arm_r_joints:i").c_str(),"tcp",false);
        //Network::connect(string("/skinManager/skin_events:o").c_str(),string("/"+local_name+"/skin_events_conv:i").c_str(),"tcp",false);
        //Network::connect(string("/wholeBodyDynamics/imu/measures:i").c_str(),string("/"+local_name+"/inertials:i").c_str(),"tcp",false);
    }
    yInfo ("Ports connected");

    init_upper();

}

void baseConversion::init_upper()
{
    //---------------------PARTS-------------------------//
    // Left_arm variables
    int jnt=16;
    F_base_Larm.resize(6,0.0);
    M_base_Larm.resize(3,0.0);
    F_ext_sens_left_arm.resize(6,0.0);
    inertial_w0.resize(3,0.0);
    inertial_dw0.resize(3,0.0);
    inertial_d2p0.resize(3,0.0);
    Wrench_base_Larm.resize(6,0.0);

    force_hand.resize(3,0.0);
    Mu_hand.resize(3,0.0);
    wrench_base_skin.resize(6,0.0);
    force_base_skin.resize(3,0.0);
    Mu_base_skin.resize(3,0.0);

    F_base.resize(3,0.0);
    M_base.resize(3,0.0);

    Wrench_gravity.resize(6,0.0);

    joints_l_arm.resize(16,0.0);
    joints_r_arm.resize(16,0.0);
    joints_l_arm_prev.resize(16,0.0);
    joints_r_arm_prev.resize(16,0.0);
    d_joints_l_arm.resize(16,0.0);
    d_joints_r_arm.resize(16,0.0);
    d_joints_l_arm_prev.resize(16,0.0);
    d_joints_r_arm_prev.resize(16,0.0);
    d2_joints_r_arm.resize(16,0.0);
    d2_joints_r_arm.resize(16,0.0);


    w0.resize(3,0.0);
    dw0.resize(3,0.0);
    d2p0.resize(3,0.0);

    // Right_arm variables
    jnt = 16;
    F_base_Rarm.resize(3,0.0);
    M_base_Rarm.resize(3,0.0);   
    F_ext_sens_right_arm.resize(6,0.0);
    Wrench_base_Rarm.resize(6,0.0);

    //std::vector<yarp::sig::Vector> normal_forces(12, yarp::sig::Vector(3));
    normal_forces.resize(12,yarp::sig::Vector(3));
    normal_forces[0] = {0.0,  0.5,  -0.866025};
    normal_forces[1] = {0.0,  1.0,  0.0};
    normal_forces[2] = {0.0,  0.5,  0.866025};
    normal_forces[3] = {0.0,  0.0,  1.0};
    normal_forces[4] = {0.0,  0.0,  -1.0};
    normal_forces[5] = {0.0,  0.5, 0.866025};
    normal_forces[6] = {0.0,  1.0,  0.0};
    normal_forces[7] = {0.0,  0.5,  -0.866025};
    normal_forces[8] = {-0.5,  0.471671,  -0.72631};
    normal_forces[9] = {-0.5,  0.866025,  0.0};
    normal_forces[10] = {-0.5,  0.471671,  0.72631};
    normal_forces[11] = {-1.0,  0.0,  0.0};

    normal_palm_init.resize(3,0.0);
    normal_palm_init[2] = 1.0;

    grav_idyn.zero();
    grav_idyn.setVal(2,-9.81);

    dofs = estimator.model().getNrOfDOFs();

    qj   = iDynTree::JointPosDoubleArray(dofs);
    dqj  = iDynTree::JointDOFsDoubleArray(dofs);
    ddqj = iDynTree::JointDOFsDoubleArray(dofs);

    qj.zero();
    dqj.zero();
    ddqj.zero();
    
}

bool baseConversion::threadInit()
{
    yInfo("threadInit: waiting for port connections... \n\n");
    // N trials to get a more accurate estimation
    for(size_t i=0; i<status_queue_size; i++)
    {
        //read joints and ft sensor
        bool ret = readAndUpdate(true,true);
        if (!ret)
        {
            yError("A problem occured during the initial readAndUpdate(), stopping... \n");
            return false;
        }
    }

    thread_status = STATUS_OK;

    return true;
}

void baseConversion::run()
{
    timestamp.update();

    thread_status = STATUS_OK;
    static int delay_check=0;
    if(!readAndUpdate(false))
    {
        delay_check++;
        yWarning ("network delays detected (%d/10)\n", delay_check);
        if (delay_check>=10)
        {
            yError ("forceConversion thread lost connection with iCubInterface.\n");
            thread_status = STATUS_DISCONNECTED;
        }
    }
    else
    {
        delay_check = 0;
    }

    /********* Implementations  *********/

    if(FT == 1){

        Force_in.resize(3,0.0);
        Mu_in.resize(3,0.0);

        // need to take into account the gravitational component and remove it!

        if (arm == "left"){
            yInfo("left");
            /*
            arm_left_bwd_1->setAng(joints_l_arm);
            arm_left_bwd_1->setDAng(d_joints_l_arm); 
            arm_left_bwd_1->setD2Ang(d2_joints_l_arm); 
            arm_left_bwd_1->computeGravity(inertial_d2p0);
            //arm_left_fwd->computeNewtonEuler(w0,dw0,d2p0,F_gravity,M_gravity); // F_gravity antes
            left_gravity->computeSensorForceMoment();
            F_gravity = left_gravity->getSensorForce();
            M_gravity = left_gravity->getSensorMoment();
            Wrench_gravity = left_gravity->getSensorForceMoment();
            */

            l_arm_index = estimator.model().getFrameIndex("torso");
            estimator.updateKinematicsFromFixedBase(qj,dqj,ddqj,l_arm_index,grav_idyn);

            unknownWrench = iDynTree::UnknownWrenchContact();
            unknownWrench.unknownType = FULL_WRENCH;

            // the position is the origin, so the contact point wrt to l_sole is zero
            unknownWrench.contactPoint.zero();

            // The fullBodyUnknowns is a class storing all the unknown external wrenches acting on a class
            fullBodyUnknowns = iDynTree::LinkUnknownWrenchContacts(estimator.model());
            fullBodyUnknowns.clear();

            fullBodyUnknowns.addNewContactInFrame(estimator.model(),l_arm_index,unknownWrench);

            // There are three output of the estimation:

            // The estimated FT sensor measurements
            estFTmeasurements = iDynTree::SensorsMeasurements(estimator.model().sensors());


            // The estimated joint torques
            estJointTorques = iDynTree::JointDOFsDoubleArray(dofs);

            // The estimated contact forces
            estContactForces = iDynTree::LinkContactWrenches(estimator.model());

            // run the estimation
            estimator.computeExpectedFTSensorsMeasurements(fullBodyUnknowns,estFTmeasurements,estContactForces,estJointTorques);

            int nrOfFTSensors = estimator.model().sensors().getNrOfSensors(iDynTree::SIX_AXIS_FORCE_TORQUE);

            for (int ftIndex = 0; ftIndex < nrOfFTSensors; ftIndex++){
                iDynTree::Wrench estimatedSensorWrench = iDynTree::Wrench();
                sens_idyn = estimator.model().sensors().getSensor(iDynTree::SIX_AXIS_FORCE_TORQUE,ftIndex);
                estFTmeasurements.getMeasurement(iDynTree::SIX_AXIS_FORCE_TORQUE,ftIndex,estimatedSensorWrench);

                iDynTree::Vector3 force = estimatedSensorWrench.getLinearVec3();
                iDynTree::Vector3 torque = estimatedSensorWrench.getAngularVec3();

                for (int k = 0; k < 3; k++){
                    Wrench_gravity[k] = force[k];

            // conversão para a base agora
            arm_left_bwd->setAng(joints_l_arm); 
            arm_left_bwd->setDAng(d_joints_l_arm);
            arm_left_bwd->setD2Ang(d2_joints_l_arm);
            bool left_done = left_FT->setSensorMeasures(F_ext_sens_left_arm);
            Wrench_base_Larm = left_FT->computeWrenchFromSensorNewtonEuler(Force_in, Mu_in); 
            
            yError() << "      Fbase0 - " << Wrench_base_Larm[0];
            yError() << "      Fbase1 - " << Wrench_base_Larm[1];
            yError() << "      Fbase2 - " << Wrench_base_Larm[2];
            broadcastData<Vector> (Wrench_base_Larm, port_FT_LA_base);
        }else if(arm  == "right"){
            //yInfo("right");

            for (int j = 0; j< qj.size(); j++){
                qj.setVal(j,joints_r_arm[j]);
            }
            
            r_arm_index = estimator.model().getFrameIndex("torso_1");
            estimator.updateKinematicsFromFixedBase(qj,dqj,ddqj,r_arm_index,grav_idyn);

            unknownWrench = iDynTree::UnknownWrenchContact();
            unknownWrench.unknownType = FULL_WRENCH;

            // the position is the origin, so the contact point wrt to l_sole is zero
            unknownWrench.contactPoint.zero();

            // The fullBodyUnknowns is a class storing all the unknown external wrenches acting on a class
            fullBodyUnknowns = iDynTree::LinkUnknownWrenchContacts(estimator.model());
            fullBodyUnknowns.clear();

            fullBodyUnknowns.addNewContactInFrame(estimator.model(),r_arm_index,unknownWrench);

            // There are three output of the estimation:

            // The estimated FT sensor measurements
            estFTmeasurements = iDynTree::SensorsMeasurements(estimator.model().sensors());


            // The estimated joint torques
            estJointTorques = iDynTree::JointDOFsDoubleArray(dofs);

            // The estimated contact forces
            estContactForces = iDynTree::LinkContactWrenches(estimator.model());

            // run the estimation
            estimator.computeExpectedFTSensorsMeasurements(fullBodyUnknowns,estFTmeasurements,estContactForces,estJointTorques);

            int nrOfFTSensors = estimator.model().sensors().getNrOfSensors(iDynTree::SIX_AXIS_FORCE_TORQUE);

            for (int ftIndex = 0; ftIndex < nrOfFTSensors; ftIndex++){
                iDynTree::Wrench estimatedSensorWrench = iDynTree::Wrench();
                sens_idyn = estimator.model().sensors().getSensor(iDynTree::SIX_AXIS_FORCE_TORQUE,ftIndex);
                estFTmeasurements.getMeasurement(iDynTree::SIX_AXIS_FORCE_TORQUE,ftIndex,estimatedSensorWrench);

                iDynTree::Vector3 force = estimatedSensorWrench.getLinearVec3();
                iDynTree::Vector3 torque = estimatedSensorWrench.getAngularVec3();

                for (int k = 0; k < 3; k++){ // depois tens de escolher se é o primeiro ou o segundo FT que sai do for
                    Wrench_gravity[k] = force[k];
                    Wrench_gravity[k + 3] = torque[k];
                }

                yError("FT new frame");
                yError() << "      Fgravity0 - " << force[0];
                yError() << "      Fgravity1 - " << force[1];
                yError() << "      Fgravity2 - " << force[2];
            }

            yError() << "      Flido0 - " << F_ext_sens_right_arm[0];
            yError() << "      Flido1 - " << F_ext_sens_right_arm[1];
            yError() << "      Flido2 - " << F_ext_sens_right_arm[2];
            
            F_ext_sens_right_arm -= Wrench_gravity;

            yError() << "      Fpossoma0 - " << F_ext_sens_right_arm[0];
            yError() << "      Fpossoma1 - " << F_ext_sens_right_arm[1];
            yError() << "      Fpossoma2 - " << F_ext_sens_right_arm[2];

            // conversão para a base agora
            arm_right_bwd->setAng(joints_r_arm); 
            arm_right_bwd->setDAng(d_joints_r_arm);
            arm_right_bwd->setD2Ang(d2_joints_r_arm);
            bool right_done = right_FT->setSensorMeasures(F_ext_sens_right_arm);
            Wrench_base_Rarm = right_FT->computeWrenchFromSensorNewtonEuler(Force_in, Mu_in); 
            
            yError() << "      Fbase0 - " << Wrench_base_Rarm[0];
            yError() << "      Fbase1 - " << Wrench_base_Rarm[1];
            yError() << "      Fbase2 - " << Wrench_base_Rarm[2];
            broadcastData<Vector> (Wrench_base_Rarm, port_FT_RA_base);
        }

    }


    /********** SKIN ******/

    if(skin == 1){ 

        force_v_index.resize(3,0.0);
        force_v_middle.resize(3,0.0);
        force_v_ring.resize(3,0.0);
        force_v_little.resize(3,0.0);
        force_v_thumb.resize(3,0.0);
        force_v_palm.resize(3,0.0);
        force_hand.resize(3,0.0);
        Mu_hand.resize(3,0.0);

        normal_index.resize(3,0.0);
        normal_middle.resize(3,0.0);
        normal_ring.resize(3,0.0);
        normal_little.resize(3,0.0);
        normal_thumb.resize(3,0.0);
        normal_palm.resize(3,0.0);

        force_index = 0.0;
        force_middle = 0.0;
        force_little = 0.0;
        force_ring = 0.0;
        force_thumb = 0.0;
        force_palm = 0.0;

        active_taxels_index = 0;
        active_taxels_middle = 0;
        active_taxels_little = 0;
        active_taxels_ring = 0;
        active_taxels_thumb = 0;
        active_taxels_palm = 0;
        
        if (arm == "left"){

            if (pressure != 0) {

                if (skin_part == 3){
                    for (int j = 0; j < taxel_list.size(); j++){
                        if (taxel_list[j] < 12){
                            active_taxels_index++;
                        }else if(taxel_list[j] < 24){
                            active_taxels_middle++;
                        }else if(taxel_list[j] < 36){
                            active_taxels_ring++;
                        }else if(taxel_list[j] < 48){
                            active_taxels_little++;
                        }else if(taxel_list[j] < 60){
                            active_taxels_thumb++;
                        }else{
                            active_taxels_palm++;
                        }
                    }

                    if (active_taxels_index > 0){ // index
                        yError("indez");
                        force_index = pressure * active_taxels_index * gain_index;
                        for (int i = 0; i < active_taxels_index; i++){
                            normal_index += normal_forces[taxel_list[i]];
                        }
                        float norm_index = sqrt(pow(normal_index[0],2) + pow(normal_index[1],2) + pow(normal_index[2],2));
                        normal_index = normal_index/norm_index;
                    }else if(active_taxels_middle > 0){ // middle
                        yError("middle");
                        force_middle = pressure * active_taxels_middle * gain_middle;
                        for (int i = active_taxels_index; i < active_taxels_middle + active_taxels_index; i++){
                            normal_middle += normal_forces[taxel_list[i]-12];
                        }
                        float norm_middle = sqrt(pow(normal_middle[0],2) + pow(normal_middle[1],2) + pow(normal_middle[2],2));
                        normal_middle = normal_middle/norm_middle;
                    }else if(active_taxels_ring > 0){ // ring
                        force_ring = pressure * active_taxels_ring * gain_ring;
                        for (int i = active_taxels_index + active_taxels_middle; i < active_taxels_middle + active_taxels_index + active_taxels_ring; i++){
                            normal_ring += normal_forces[taxel_list[i]-24];
                        }
                        float norm_ring = sqrt(pow(normal_ring[0],2) + pow(normal_ring[1],2) + pow(normal_ring[2],2));
                        normal_ring = normal_ring/norm_ring;
                    }else if(active_taxels_little > 0){ // little
                        force_little = pressure * active_taxels_little * gain_little;
                        for (int i = active_taxels_index + active_taxels_middle + active_taxels_ring; i < active_taxels_middle + active_taxels_index + active_taxels_ring + active_taxels_little; i++){
                            normal_little += normal_forces[taxel_list[i]-36];
                        }
                        float norm_little = sqrt(pow(normal_little[0],2) + pow(normal_little[1],2) + pow(normal_little[2],2));
                        normal_little = normal_little/norm_little;
                    }else if(active_taxels_thumb > 0){ // if true is the thumb
                        force_thumb = pressure * active_taxels_thumb * gain_thumb;
                        for (int i = active_taxels_index + active_taxels_middle + active_taxels_ring + active_taxels_little; i < active_taxels_middle + active_taxels_index + active_taxels_ring + active_taxels_little + active_taxels_thumb; i++){
                            normal_thumb += normal_forces[taxel_list[i]-48];
                        }
                        float norm_thumb = sqrt(pow(normal_thumb[0],2) + pow(normal_thumb[1],2) + pow(normal_thumb[2],2));
                        normal_thumb = normal_thumb/norm_thumb;
                    }else if(active_taxels_palm > 0){
                        force_palm = pressure * active_taxels_palm * gain_palm;
                    }  
                }
            }

            // Use std::accumulate to sum the elements
            yError("cheguei vivo aqui");
            if (force_palm != 0){
                force_v_palm = force_palm * normal_palm_init;
            }
            if (force_ring != 0){
                force_v_ring = force_ring * normal_ring;
            }
            if (force_thumb != 0){
                force_v_thumb = force_thumb * normal_thumb;
            }
            if (force_little != 0){
                force_v_little = force_little * normal_little;
            }
            if (force_index != 0){
                force_v_index = force_index * normal_index;
            }
            if (force_middle != 0){
                force_v_middle = force_middle * normal_middle;
            }

            yError("somar forças");

            yError() << "      FIndex - " << force_index;
            yError() << "      FMiddle - " << force_middle;
            yError() << "      FRing - " << force_ring;
            yError() << "      FLittle - " << force_little;
            yError() << "      FThumb - " << force_thumb;
            yError() << "      FPalm - " << force_palm;

            yError("algum problema aqui??");
            yError() << "      FIndex_size - " << size(force_v_index);
            yError() << "      FMiddle_size - " << size(force_v_middle);
            yError() << "      FRing_size - " << size(force_v_ring);
            yError() << "      FLittle_size - " << size(force_v_little);
            yError() << "      FThumb_size - " << size(force_v_thumb);
            yError() << "      FPalm_size - " << size(force_v_palm);


            Vector prev = force_v_index + force_v_middle + force_v_ring + force_v_little + force_v_thumb + force_v_palm;

            yError() << "      prev_size - " << size(prev);

            yError("bora");

            force_hand[0] = prev[0];
            force_hand[1] = prev[1];
            force_hand[2] = prev[2];
            Mu_hand[0] = 0.0;
            Mu_hand[1] = 0.0;
            Mu_hand[2] = 0.0;

            arm_left_bwd->setAng(joints_l_arm);

            left_skin->BackwardWrenchFromEnd(force_hand,Mu_hand); // Atenção que aqui para já também não há compensação de gravidade

            left_skin->getWrenchBase(force_base_skin, Mu_base_skin);

            for (size_t i = 0; i < 3; i++) {
                wrench_base_skin[i] = force_base_skin[i];
            }

            for (size_t i = 0; i < 3; i++) {
                wrench_base_skin[i+3] = Mu_base_skin[i];
            }

            // dar broadcast depois!! Falta criar porto!
            broadcastData<Vector> (wrench_base_skin, port_wrench_l_skin);


        }else if(arm == "right"){

            if (pressure != 0) {

                yError("error??");

                if (skin_part == 4){
                    for (int j = 0; j < taxel_list.size(); j++){
                        if (taxel_list[j] < 12){
                            active_taxels_index++;
                        }else if(taxel_list[j] < 24){
                            active_taxels_middle++;
                        }else if(taxel_list[j] < 36){
                            active_taxels_ring++;
                        }else if(taxel_list[j] < 48){
                            active_taxels_little++;
                        }else if(taxel_list[j] < 60){
                            active_taxels_thumb++;
                        }else{
                            active_taxels_palm++;
                        }
                    }

                    if (active_taxels_index > 0){ // index
                        yError("indez");
                        force_index = pressure * active_taxels_index * gain_index;
                        for (int i = 0; i < active_taxels_index; i++){
                            normal_index += normal_forces[taxel_list[i]];
                        }
                        float norm_index = sqrt(pow(normal_index[0],2) + pow(normal_index[1],2) + pow(normal_index[2],2));
                        normal_index = normal_index/norm_index;
                    }if(active_taxels_middle > 0){ // middle
                        yError("middle");
                        force_middle = pressure * active_taxels_middle * gain_middle;
                        for (int i = active_taxels_index; i < active_taxels_middle + active_taxels_index; i++){
                            normal_middle += normal_forces[taxel_list[i]-12];
                        }
                        float norm_middle = sqrt(pow(normal_middle[0],2) + pow(normal_middle[1],2) + pow(normal_middle[2],2));
                        normal_middle = normal_middle/norm_middle;
                    }if(active_taxels_ring > 0){ // ring
                        force_ring = pressure * active_taxels_ring * gain_ring;
                        for (int i = active_taxels_index + active_taxels_middle; i < active_taxels_middle + active_taxels_index + active_taxels_ring; i++){
                            normal_ring += normal_forces[taxel_list[i]-24];
                        }
                        float norm_ring = sqrt(pow(normal_ring[0],2) + pow(normal_ring[1],2) + pow(normal_ring[2],2));
                        normal_ring = normal_ring/norm_ring;
                    }if(active_taxels_little > 0){ // little
                        force_little = pressure * active_taxels_little * gain_little;
                        for (int i = active_taxels_index + active_taxels_middle + active_taxels_ring; i < active_taxels_middle + active_taxels_index + active_taxels_ring + active_taxels_little; i++){
                            normal_little += normal_forces[taxel_list[i]-36];
                        }
                        float norm_little = sqrt(pow(normal_little[0],2) + pow(normal_little[1],2) + pow(normal_little[2],2));
                        normal_little = normal_little/norm_little;
                    }if(active_taxels_thumb > 0){ // if true is the thumb
                        force_thumb = pressure * active_taxels_thumb * gain_thumb;
                        for (int i = active_taxels_index + active_taxels_middle + active_taxels_ring + active_taxels_little; i < active_taxels_middle + active_taxels_index + active_taxels_ring + active_taxels_little + active_taxels_thumb; i++){
                            normal_thumb += normal_forces[taxel_list[i]-48];
                        }
                        float norm_thumb = sqrt(pow(normal_thumb[0],2) + pow(normal_thumb[1],2) + pow(normal_thumb[2],2));
                        normal_thumb = normal_thumb/norm_thumb;
                    }if(active_taxels_palm > 0){
                        force_palm = pressure * active_taxels_palm * gain_palm;
                    }  
                }    
            }

            if (force_palm != 0){
                force_v_palm = force_palm * normal_palm;
            }
            if (force_ring != 0){
                force_v_ring = force_ring * normal_ring;
            }
            if (force_thumb != 0){
                force_v_thumb = force_thumb * normal_thumb;
            }
            if (force_little != 0){
                force_v_little = force_little * normal_little;
            }
            if (force_index != 0){
                force_v_index = force_index * normal_index;
            }
            if (force_middle != 0){
                force_v_middle = force_middle * normal_middle;
            }

            yError("somar forças");

            yError() << "      FIndex - " << force_index;
            yError() << "      FMiddle - " << force_middle;
            yError() << "      FRing - " << force_ring;
            yError() << "      FLittle - " << force_little;
            yError() << "      FThumb - " << force_thumb;
            yError() << "      FPalm - " << force_palm;


            yError() << "      middle vector x - " << force_v_middle[0];
            yError() << "      middle vector y - " << force_v_middle[1];
            yError() << "      middle vector z - " << force_v_middle[2];

            Vector prev = force_v_index + force_v_middle + force_v_ring + force_v_little + force_v_thumb + force_v_palm;

            yError() << "      prev x - " << prev[0];
            yError() << "      prev y - " << prev[1];
            yError() << "      prev z - " << prev[2];

            yError("bora");

            force_hand[0] = prev[0];
            force_hand[1] = prev[1];
            force_hand[2] = prev[2];
            Mu_hand[0] = 0.0;
            Mu_hand[1] = 0.0;
            Mu_hand[2] = 0.0;

            arm_right_bwd->setAng(joints_r_arm);

            right_skin->BackwardWrenchFromEnd(force_hand,Mu_hand); // Atenção que aqui para já também não há compensação de gravidade

            right_skin->getWrenchBase(force_base_skin, Mu_base_skin);

            yError() << "      Fbase x - " << force_base_skin[0];
            yError() << "      Fbase y - " << force_base_skin[1];
            yError() << "      Fbase z - " << force_base_skin[2];

            for (size_t i = 0; i < 3; i++) {
                wrench_base_skin[i] = force_base_skin[i];
            }

            yError("made everything");

            for (size_t i = 0; i < 3; i++) {
                wrench_base_skin[i+3] = Mu_base_skin[i];
            }

            broadcastData<Vector> (wrench_base_skin, port_wrench_r_skin);
        
        }

        /*
        Importante testar tocar nas mãos do iCub em 2 diferentes tactile sensors ao mesmo tempo e ver oq sai
        Ele não terá só um dedo a tocar no objeto
        Importante distinguir os diferentes vetores dos dedos!!
        Somar todos os vetores para um só no hand frame
        Não esquecer que eles tem de ser wrenches
        Podemos considerar que os momentos são 0 pois os objetos não estarão em movimento na mão idealmente
        Converter depois claro para a base frame
        */
    }

}

bool baseConversion::readAndUpdate(bool waitMeasure, bool _init)
{
    bool b = true;
    
    if (FT == 1){
        if(arm == "left"){
            // NO SIMULADOR
            yError("vamos ler!!");
            Vector *FT_LA = port_FT_LA->read(waitMeasure);
            if (FT_LA != nullptr){
                F_ext_sens_left_arm[0] = (*FT_LA)[0];
                F_ext_sens_left_arm[1] = (*FT_LA)[1];
                F_ext_sens_left_arm[2] = (*FT_LA)[2];
                F_ext_sens_left_arm[3] = (*FT_LA)[3];
                F_ext_sens_left_arm[4] = (*FT_LA)[4];
                F_ext_sens_left_arm[5] = (*FT_LA)[5];
            /*
            double timestamp;
            Vector FT_LA;
            m_left_arm_FT->getSixAxisForceTorqueSensorMeasure(0, FT_LA, timestamp);
            if (FT_LA.size() > 0){
                yInfo("after left");
                F_ext_sens_left_arm[0] = (FT_LA)[0];
                F_ext_sens_left_arm[1] = (FT_LA)[1];
                F_ext_sens_left_arm[2] = (FT_LA)[2];
                F_ext_sens_left_arm[3] = (FT_LA)[3];
                F_ext_sens_left_arm[4] = (FT_LA)[4];
                F_ext_sens_left_arm[5] = (FT_LA)[5];
                yInfo("Read");
                yError() << "      Fx - " << F_ext_sens_left_arm[0];
                yError() << "      Fy - " << F_ext_sens_left_arm[1];
                yError() << "      Fz - " << F_ext_sens_left_arm[2];
            */
            } else {
                yError("Failed to read data from port_FT_LA");
                F_ext_sens_left_arm.zero();
            }
        }else if(arm == "right"){
            // NO SIMULADOR
            Vector *FT_RA = port_FT_RA->read(waitMeasure);
            if (FT_RA != nullptr){
                yInfo("after right");
                F_ext_sens_right_arm[0] = (*FT_RA)[0];
                F_ext_sens_right_arm[1] = (*FT_RA)[1];
                F_ext_sens_right_arm[2] = (*FT_RA)[2];
                F_ext_sens_right_arm[3] = (*FT_RA)[3];
                F_ext_sens_right_arm[4] = (*FT_RA)[4];
                F_ext_sens_right_arm[5] = (*FT_RA)[5];
            /*
            double timestamp;
            Vector FT_RA;
            m_right_arm_FT->getSixAxisForceTorqueSensorMeasure(0, FT_RA, timestamp);
            if (FT_RA.size() > 0){
                yInfo("after right");
                F_ext_sens_right_arm[0] = (FT_RA)[0];
                F_ext_sens_right_arm[1] = (FT_RA)[1];
                F_ext_sens_right_arm[2] = (FT_RA)[2];
                F_ext_sens_right_arm[3] = (FT_RA)[3];
                F_ext_sens_right_arm[4] = (FT_RA)[4];
                F_ext_sens_right_arm[5] = (FT_RA)[5];
            */
            } else {
                yError("Failed to read data from port_FT_RA");
                F_ext_sens_right_arm.zero();
            }
        }else{
            yInfo("I am not a octopus. I do not have any more arms...");
        }  

    }

    if (arm == "left"){
        yarp::sig::VectorOf<double>* temp = port_joints_l_arm->read(waitMeasure);
        if (temp != nullptr) {
            joints_l_arm_prev = joints_l_arm;
            d_joints_l_arm_prev = d_joints_l_arm;
            joints_l_arm = *temp;
            d_joints_l_arm = joints_l_arm - joints_l_arm_prev;
            d2_joints_l_arm = d_joints_l_arm - d_joints_l_arm_prev;
            /*
            yError() << "      Joint 0 - " << joints_l_arm[0];
            yError() << "      Joint 1 - " << joints_l_arm[1];
            yError() << "      Joint 2 - " << joints_l_arm[2];
            yError() << "      DJoint 0 - " << d_joints_l_arm[0];
            yError() << "      DJoint 1 - " << d_joints_l_arm[1];
            yError() << "      DJoint 2 - " << d_joints_l_arm[2];
            yError() << "      D2Joint 0 - " << d2_joints_l_arm[0];
            yError() << "      D2Joint 1 - " << d2_joints_l_arm[1];
            yError() << "      D2Joint 2 - " << d2_joints_l_arm[2];
            */
        }
    } else if(arm == "right"){
        yarp::sig::VectorOf<double>* temp = port_joints_r_arm->read(waitMeasure);
        
            if (temp != nullptr) {
                joints_r_arm_prev = joints_r_arm;
                d_joints_r_arm_prev = d_joints_r_arm;
                joints_r_arm = *temp;
                d_joints_r_arm = joints_r_arm - joints_r_arm_prev;
                d2_joints_r_arm = d_joints_r_arm - d_joints_r_arm_prev;
            }
    }

    inertial_d2p0[0] = 0.0;
    inertial_d2p0[1] = 0.0;
    inertial_d2p0[2] = 9.81;  

    
    /*

    Vector *inertial = port_inertials->read(waitMeasure);
    if (waitMeasure) yInfo("done. \n");

    int sz = 0;
    if(inertial!=nullptr)
    {
        inertial_d2p0[0] = (*inertial)[0];
        inertial_d2p0[1] = (*inertial)[1];
        inertial_d2p0[2] = (*inertial)[2];  
        inertial_w0[0] =  (*inertial)[3]*CTRL_DEG2RAD;
        inertial_w0[1] =  (*inertial)[4]*CTRL_DEG2RAD;
        inertial_w0[2] =  (*inertial)[5]*CTRL_DEG2RAD;
        inertial_dw0 = this->eval_domega(inertial_w0);
    }
    
    */


    // Skin Contacts
    if(skin == 1){
        addSkinContacts();
    }

    return b;
}

void baseConversion::addSkinContacts()
{

    //INIT
    pressure = 0.0;
    skin_part = 1;
    active_taxels = 0;
    normal_directions.resize(3,0.0);
    positions.resize(3,0.0);


    skinContactList *scl = port_skin_events->read(false);
    yError("li");
    if(scl)
    {
        yInfo("entrou");
        skinContactsTimestamp = Time::now();
        if(scl->empty() && !default_ee_cont)   // if no skin contacts => leave the old contacts but reset pressure and contact list
        {
            for(auto & skinContact : skinContacts)
            {
                skinContact.setPressure(0.0);
                skinContact.setActiveTaxels(0);
            }
            return;
        }

        map<BodyPart, skinContactList> contactsPerBp = scl->splitPerBodyPart();
        // if there are more than 1 contact and less than 10 taxels are active then suppose zero moment
        yInfo("split body part");
        for(auto & it : contactsPerBp){
        yInfo("for bp");
            if(it.second.size()>1){
                for(auto & c : it.second){
                    if(c.getActiveTaxels()<10){
                        c.fixMoment();
                    }
                }
            }
        }
        icub->upperTorso->clearContactList();
        //icub->upperTorso->leftSensor->addContacts(contactsPerBp[LEFT_ARM].toDynContactList());
        icub->upperTorso->rightSensor->addContacts(contactsPerBp[RIGHT_ARM].toDynContactList());
        skinContacts = contactsPerBp[RIGHT_ARM];
        //skinContacts = contactsPerBp[LEFT_ARM];
        //skinContacts.insert(skinContacts.end(), contactsPerBp[RIGHT_ARM].begin(), contactsPerBp[RIGHT_ARM].end());

        yInfo("bue cenas");
        yError() << "      skin contacts size - " << skinContacts.size();
        for(auto & skinContact : skinContacts)
        {   
            yInfo("engradeieis");
            active_taxels = skinContact.getActiveTaxels();
            yInfo("engradeieis 1");
            // importante aqui talvez ver outras coisas também
            // like skin part or activated taxels!!
            pressure = skinContact.getPressure();
            normal_directions = skinContact.getNormalDir();
            positions = skinContact.getGeoCenter();
            skin_part = skinContact.getSkinPart();
            taxel_list = skinContact.getTaxelList();
            yInfo("all info");
            yError() << "      Pressure - " << pressure;
            yError() << "      Active Taxels - " << active_taxels;
            yError() << "      Positions - " << positions[0];
            yError() << "      skin_part - " << skin_part;
            yError() << "      Taxels 0 - " << taxel_list[0];
            yError() << "      Taxels 1 - " << taxel_list[1];
            yError() << "      Taxels 2 - " << taxel_list[2];

        } 
            
    }
    else if(Time::now()-skinContactsTimestamp>SKIN_EVENTS_TIMEOUT && skinContactsTimestamp!=0.0)
    {
        // if time is up, remove all the contacts
        icub->upperTorso->clearContactList();
        skinContacts.clear();
    }

    yInfo("acabouuuuuuuuu");

}

Vector baseConversion::eval_domega(const Vector &x)
{
    AWPolyElement el;
    el.data=x;
    el.time=Time::now();

    return InertialEst->estimate(el);
}

template <class T> void baseConversion::broadcastData(T& _values, BufferedPort<T> *_port)
{
    if (_port && _port->getOutputCount()>0)
    {
        _port->setEnvelope(this->timestamp);
        _port->prepare()  = _values ;
        _port->write();
    }
}

void baseConversion::threadRelease()
{

    // falta fechar todos os outros portos!

    yInfo( "ForceConversion: Closing FT_RA port\n");
    closePort(port_FT_RA);
    yInfo( "ForceConversion: Closing FT_LA port\n");
    closePort(port_FT_LA);
    yInfo( "ForceConversion: Closing FT_RA_base port\n");
    closePort(port_FT_RA_base);
    yInfo( "ForceConversion: Closing FT_LA_base port\n");
    closePort(port_FT_LA_base);
    yInfo( "ForceConversion: Closing skin_contacts port\n");
    closePort(port_skin_events);

}

void baseConversion::closePort(Contactable *_port)
{
    if (_port)
    {
        _port->interrupt();
        _port->close();

        delete _port;
        _port = nullptr;
    }
}

void baseConversion::update_inertial_data(const Vector &inertial)
{
    inertial_d2p0[0] = (inertial)[0];
    inertial_d2p0[1] = (inertial)[1];
    inertial_d2p0[2] = (inertial)[2];
    inertial_w0 [0] =  (inertial)[3]*CTRL_DEG2RAD;
    inertial_w0 [1] =  (inertial)[4]*CTRL_DEG2RAD;
    inertial_w0 [2] =  (inertial)[5]*CTRL_DEG2RAD;
    inertial_dw0 = this->eval_domega(inertial_w0);
}

