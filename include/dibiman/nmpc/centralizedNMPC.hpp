#ifndef __dibiman_centralized_nmpc__
#define __dibiman_centralized_nmpc__

#include <casadi/casadi.hpp>

#include "dibiman/nmpc/baseNMPC.hpp"

namespace dibiman {
    class centralizedNMPC: public baseNMPC {
        casadi::Opti optimizer;

        /* right arm variables */
        std::vector<casadi::MX> Xr;
        std::vector<casadi::MX> Ar;
        std::vector<casadi::MX> Ur;

        /* right hand parameters */
        casadi::MX x0_r;
        casadi::MX xref_r;
        casadi::MX uref_r;

        /* left hand variables */
        std::vector<casadi::MX> Xl;
        std::vector<casadi::MX> Al;
        std::vector<casadi::MX> Ul;

        /* left hand parameters */
        casadi::MX x0_l;
        casadi::MX xref_l;
        casadi::MX uref_l;

        public:
            centralizedNMPC(
                    const std::string& model_path,
                    const std::vector<std::vector<std::string>> & joints_to_use,
                    const std::vector<std::string>& list_ee_frame_names,
                    const std::vector<std::string>& ids
                    );

            void initSX_OCP(void);

            void createOCP(void) override;

            std::map<std::string, casadi::DM> solve(const std::map<std::string, baseNMPC::armData>& manip_data) override;
    };
};
  
#endif //__dibiman_centralized_nmpc__
