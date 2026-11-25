#pragma once

#ifndef __dibiman_trajectory_generator__
#define __dibiman_trajectory_generator__

#include "pinocchio/spatial.hpp"
#include "pinocchio/algorithm/frames.hpp"

namespace dibiman {
    class TrajectoryGenerator{
        pinocchio::SE3 initial_X;
        pinocchio::SE3 final_X;
        double final_T;
        double elapsed_time;

        /* quintic trajectory interpolation constants */
        static constexpr double a5 = 6;
        static constexpr double a4 = -15;
        static constexpr double a3 = 10;

        inline double st(const double&);
        inline double d_st(const double&);

        public:
        pinocchio::Motion s_twist(const double&);
        void step(const double&);
        TrajectoryGenerator();
        void setData(const pinocchio::SE3 &, const pinocchio::SE3 &, const double &);
    };
}

#endif //__dibiman_trajectory_generator__
