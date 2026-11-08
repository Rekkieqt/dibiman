int main()
{
    rnea(model, data, q, v, a);
    Force f_model = computeSupportedForceByFrame(model, data, ft_sensor_frame_ID);
    Force f_ext_sensor = f_meas_in_sensor_frame - f_model;
}
