    // inline double get_s(const double & t, const double & Tf)
    // {
    //     double x = t / Tf;
    //     double x3 = x * x * x;
    //     return a3 * x3 + a4 * x3 * x + a5 * x3 * x * x;
    // }

    // inline double get_ds(const double & t, const double & Tf)
    // {
    //     double x = t / Tf;
    //     double x2 = x * x;
    //     double res = 3 * a3 * x2 + 4 * a4 * x2 * x + 5 * a5 * x2 * x2;
    //     return res * (1.0)/Tf;
    // }

    // inline pinocchio::Motion twist_ref(const fatrop::Index k)
    // {
    //     double s = get_s(k * dt_, Tf);
    //     double ds = get_ds(k * dt_, Tf);
    //     pinocchio::Motion nu = pinocchio::log6(Xi.actInv(Xf));
    //     pinocchio::SE3 Xs = Xi * pinocchio::exp6(nu * s);
    //     return Xs.act(nu) * ds;
    // }
