#ifndef FORCE_CONV
#define FORCE_CONV

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>
#include <iCub/ctrl/math.h>
#include <iCub/ctrl/adaptWinPolyEstimator.h>
#include <iCub/iDyn/iDyn.h>
#include <iCub/iDyn/iDynBody.h>
#include <iCub/iKin/iKinFwd.h>
#include <iCub/iDyn/iDynTransform.h>
#include <iCub/skinDynLib/Taxel.h>
#include <iCub/skinDynLib/skinContactList.h>

#include <iDynTree/Utils.h>
// iDynTree includes
#include <iDynTree/Wrench.h>
#include <iDynTree/MatrixFixSize.h>
#include <iDynTree/KinDynComputations.h>
#include <iDynTree/ExtWrenchesAndJointTorquesEstimator.h>
#include <iDynTree/skinDynLibConversions.h>
#include <iDynTree/Sensors.h>
#include <iDynTree/EigenHelpers.h>
#include <iDynTree/YARPConversions.h>
#include <iDynTree/ContactStateMachine.h>
// #include <iDynTree/Estimation/KalmanFilter.h>
#include <iDynTree/Indices.h>

#include <iDynTree/ModelLoader.h>
#include <iDynTree/KinDynComputations.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <list>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;
using namespace yarp::dev;
using namespace iCub::ctrl;
using namespace iCub::iDyn;
using namespace iCub::iKin;
using namespace iCub::skinDynLib;
using namespace std;

constexpr int8_t MAX_JN = 12;
constexpr int8_t MAX_FILTER_ORDER = 6;
constexpr float_t SKIN_EVENTS_TIMEOUT = 0.2F;     // max time (in sec) a contact is kept without reading anything from the skin events port

enum thread_status_enum {STATUS_OK=0, STATUS_DISCONNECTED};
enum calib_enum {CALIB_ALL=0, CALIB_ARMS, CALIB_LEGS, CALIB_FEET};

class baseConversion: public PeriodicThread
{
public:
    bool       com_enabled;
    bool       com_vel_enabled;
    bool       dummy_ft;
    bool       w0_dw0_enabled;
    bool       dumpvel_enabled;
    bool       auto_drift_comp;
    bool       default_ee_cont;

private:

    
    string      robot_name;
    string      local_name;
    bool        autoconnect;
    int FT;
    int skin;
    string arm;
    version_tag icub_type;

    float gain_thumb;
    float gain_index;
    float gain_middle;
    float gain_ring;
    float gain_little;
    float gain_palm;

    PolyDriver *ddAL;
    PolyDriver *ddAR;
    //IEncoders  *iencs;
    IInteractionMode *iint_arm_left;
    IInteractionMode *iint_arm_right;
    IControlMode *icmd_arm_left;
    IControlMode *icmd_arm_right;
    ISixAxisForceTorqueSensors* m_left_arm_FT{nullptr};
    ISixAxisForceTorqueSensors* m_right_arm_FT{nullptr};

    Vector eval_domega(const Vector &x);

    PolyDriver *ddT;
    IInteractionMode *iint_torso;
    IControlMode *icmd_torso;
    yarp::os::Stamp timestamp;
    AWLinEstimator  *InertialEst;

    string part;
    size_t status_queue_size;

    const  long double zero_sens_tolerance;
    double skinContactsTimestamp;

    iCubWholeBody *icub;
    //CubWholeBody *icub_sens;

    iDynSensorArm *left_FT;
    iDynSensorArm *right_FT;
    OneChainNewtonEuler *left_skin;
    OneChainNewtonEuler *right_skin;
    iDynInvSensorArm *left_gravity;
    iDynInvSensorArm *right_gravity;
    iCubArmDyn *arm_right_bwd_1;
    iCubArmDyn *arm_left_bwd_1;
    iCubArmDyn *arm_right_bwd;
    iCubArmDyn *arm_left_bwd;

    bool first;
    thread_status_enum thread_status;

    BufferedPort<Vector> *port_FT_RA;
    BufferedPort<Vector> *port_FT_LA;
    BufferedPort<Vector> *port_FT_LA_base;
    BufferedPort<Vector> *port_FT_RA_base;
    BufferedPort<Vector> *port_wrench_l_skin;
    BufferedPort<Vector> *port_r_gravity;
    BufferedPort<Vector> *port_l_gravity;
    BufferedPort<Vector> *port_wrench_r_skin;
    BufferedPort<Vector> *port_joints_l_arm;
    BufferedPort<Vector> *port_joints_r_arm;
    BufferedPort<Vector> *port_inertials;
    BufferedPort<skinContactList> *port_skin_events;

    iCub::skinDynLib::skinContactList skinContacts;
    iCub::skinDynLib::dynContactList dynContacts;
    iCub::skinDynLib::Taxel Taxels;

    iDynTree::ExtWrenchesAndJointTorquesEstimator estimator;

    Vector F_ext_sens_right_arm, F_ext_sens_left_arm;       // external wrench seen at the F/T sensors
    Vector F_skin_Rarm, F_skin_Larm;
    Vector F_base_skin_R, F_base_skin_L;
    Vector F_base_Rarm, F_base_Larm;
    Vector M_base_Rarm, M_base_Larm;
    Vector FT_LA, FT_RA;
    Vector Force_in, Mu_in;
    Vector inertial_w0;
    Vector inertial_dw0;
    Vector inertial_d2p0;
    float pressure;
    Vector normal_directions;
    Vector positions;
    int skin_part;
    int active_taxels;
    Vector Force;
    Vector Wrench_base_Rarm, Wrench_base_Larm;
    Vector F_base, M_base;
    Vector F_gravity, M_gravity;
    Vector joints_l_arm, joints_r_arm;
    Vector joints_l_arm_prev, joints_r_arm_prev;
    Vector d_joints_l_arm, d_joints_r_arm;
    Vector d_joints_l_arm_prev, d_joints_r_arm_prev;
    Vector d2_joints_l_arm, d2_joints_r_arm;
    Vector w0,dw0,d2p0;
    Vector Wrench_gravity;
    iDynTree::VectorFixSize<3> grav_idyn;
    iDynTree::VectorDynSize qj, dqj, ddqj;
    int dofs;

    iDynTree::Sensor *sens_idyn;
    iDynTree::SensorsMeasurements estFTmeasurements;
    iDynTree::LinkUnknownWrenchContacts fullBodyUnknowns;
    iDynTree::JointDOFsDoubleArray estJointTorques;
    iDynTree::LinkContactWrenches estContactForces;
    iDynTree::UnknownWrenchContact unknownWrench;
    iDynTree::UnknownWrenchContactType FULL_WRENCH;
    iDynTree::FrameIndex l_arm_index;
    iDynTree::FrameIndex r_arm_index;

    float force_palm;
    float force_index;
    float force_thumb;
    float force_middle;
    float force_ring;
    float force_little;
    Vector force_v_palm;
    Vector force_v_index;
    Vector force_v_thumb;
    Vector force_v_middle;
    Vector force_v_ring;
    Vector force_v_little;
    Vector force_hand;
    Vector force_base_skin;
    Vector wrench_base_skin;
    Vector Mu_hand;
    Vector Mu_base_skin;
    Vector normal_index;
    Vector normal_thumb;
    Vector normal_ring;
    Vector normal_little;
    Vector normal_middle;
    Vector normal_palm;

    std::vector<unsigned int>  taxel_list;
    float active_taxels_index;
    float active_taxels_middle;
    float active_taxels_ring;
    float active_taxels_thumb;
    float active_taxels_little;
    float active_taxels_palm;

    std::vector<yarp::sig::Vector> normal_forces;
    Vector normal_palm_init;

    void init_upper();
    void setUpperMeasure(bool _init=false);

public:
    baseConversion(int _rate, int _FT, int _skin, string _arm, string _robot_name, string _local_name, bool _autoconnect=false,
                    ISixAxisForceTorqueSensors* m_left_arm_FT = nullptr, ISixAxisForceTorqueSensors* m_right_arm_FT = nullptr, 
                    float _gain_index = 1.0f, float _gain_middle = 1.0f, float _gain_ring = 1.0f, float _gain_little = 1.0f, float _gain_thumb = 1.0f, float _gain_palm = 1.0f);

    inline thread_status_enum getThreadStatus()
    {
        return thread_status;
    }
    void setStiffMode();
    bool threadInit() override;
    bool readAndUpdate(bool waitMeasure=false, bool _init=false);
    void run() override;
    template <class T> void broadcastData(T& _values, BufferedPort<T> *_port);
    void threadRelease() override;
    void closePort(Contactable *_port);
    void update_inertial_data(const Vector &inertial);
    void addSkinContacts();
};

#endif
