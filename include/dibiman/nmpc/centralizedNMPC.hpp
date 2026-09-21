#ifndef __dibiman_centralized_nmpc__
#define __dibiman_centralized_nmpc__

#include <casadi/casadi.hpp>

#include "dibiman/nmpc/baseNMPC.hpp"
#include "dibiman/utils/utils.hpp"

namespace dibiman {
    class centralizedNMPC: public baseNMPC {

        public:
            centralizedNMPC(
                    const std::string& model_path,
                    const std::vector<std::vector<std::string>> & joints_to_use,
                    const std::vector<std::string>& list_ee_frame_names,
                    const std::vector<std::string>& ids
                    );

            void initSX_OCP(void);

            void createOCP(void) override;

            std::map<std::string, casadi::DM> solve(const std::map<std::string, armTargetParams>&) override;
    };
};
  
#endif //__dibiman_centralized_nmpc__
