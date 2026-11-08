
int main()
{
    using namespace pinocchio;

    /* d(Jv)/dq = v_partial_dq (6 x nv) */
    /* d(Jv)/dv = v_partial_dv (6 x nv) */
    computeForwardKinematicsDerivatives(model, data, q, v);
    updateFramePlacements(model, data);
    getFrameVelocityDerivatives(model, data, frame_id, rf,
            v_partial_dq,
            v_partial_dv);
    
    /* Jres = (6 x (nv*2)) */
    // Jres.leftCols(nv),
    // Jres.rightCols(nv));

    Eigen::Vector6 spatial_twist = getFrameVelocity(model, data, frame_id, rf).toVector();
    Eigen::Vector6 ref_twist = (...);
    Eigen::Vector6 twist_residual = spatial_twist - ref_twist;

    return 0;
}
