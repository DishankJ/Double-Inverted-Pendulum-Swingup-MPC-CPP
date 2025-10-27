#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <mpc/NLMPC.hpp>

constexpr int num_states = 6;
constexpr int num_output = 6;
constexpr int num_inputs = 1;
constexpr int pred_hor = 30;
constexpr int ctrl_hor = pred_hor;
constexpr int ineq_c = (pred_hor + 1) * (2*num_states + 2);
constexpr int eq_c = 0;
double ts = 0.03;

double m0 = 0.6;
double m1 = 0.2;
double m2 = 0.2;
double L1 = 0.5;
double L2 = 0.5;
double l1 = L1/2;
double l2 = L2/2;
double g = 9.8;
double I1 = (1.0/12.0)*m1*pow(L1,2);
double I2 = (1.0/12.0)*m2*pow(L2,2);

double A = m0 + m1 + m2;
double B = m1 * l1 + m2 * L1;
double C = m2 * l2;
double D = I1 + m1 * pow(l1,2) + m2 * pow(L1,2);
double E = m2 * L1 * l2;
double F = I2 + m2 * pow(l2,2);

void doubleInvPendulumDynamics(mpc::cvec<num_states> &dx, const mpc::cvec<num_states> &x, const mpc::cvec<num_inputs> &u)
{
    dx(0) = x(3);
    dx(1) = x(4);
    dx(2) = x(5);

    // dx(3) = (900.0*pow(cos(x(1)-x(2)),2)*u(0)-1600.0*u(0)-33.75*(sin(x(1)-2*x(2))+sin(3*x(1)-2*x(2)))*pow(x(4),2)+11.25*(sin(2*x(1)-3*x(2))+sin(2*x(1)-x(2)))*pow(x(5),2)-90.0*sin(x(1)-x(2))*cos(x(1))*pow(x(5),2)+120.0*sin(x(1)-x(2))*cos(x(2))*pow(x(4),2)+135.0*sin(x(1))*pow(cos(x(1)-x(2)),2)*pow(x(4),2)-2646.0*sin(x(1))*cos(x(1)-x(2))*cos(x(2))-240.0*sin(x(1))*pow(x(4),2)+2646.0*sin(2*x(1))+45.0*sin(x(2))*pow(cos(x(1)-x(2)),2)*pow(x(5),2)-2646.0*sin(x(2))*cos(x(1)-x(2))*cos(x(1))-80.0*sin(x(2))*pow(x(5),2)+1176.0*sin(2*x(2)))/(315.0*cos(2*x(1)-2*x(2))+135.0*cos(2*x(1))-15.0*cos(2*x(2))-895.0);
    // dx(4) = (-450.0*u(0)*cos(x(1)-2*x(2))+1350.0*u(0)*cos(x(1))-3087.0*sin(x(1)-2*x(2))+232.5*sin(x(1)-x(2))*pow(x(5),2)+22.5*sin(x(1)+x(2))*pow(x(5),2)+157.5*sin(2*x(1)-2*x(2))*pow(x(4),2)-11907.0*sin(x(1))+67.5*sin(2*x(1))*pow(x(4),2))/(157.5*cos(2*x(1)-2*x(2))+67.5*cos(2*x(1))-7.5*cos(2*x(2))-447.5);
    // dx(5) = (-2.14285714285714*u(0)*cos(2*x(1)-x(2))+1.66666666666667*u(0)*cos(x(2))-1.29761904761905*sin(x(1)-x(2))*pow(x(4),2)-0.035714285714286*sin(x(1)+x(2))*pow(x(4),2)-0.25*sin(2*x(1)-2*x(2))*pow(x(5),2)+14.7*sin(2*x(1)-x(2))-10.0333333333333*sin(x(2))-0.0119047619047619*sin(2*x(2))*pow(x(5),2))/(0.25*cos(2*x(1)-2*x(2))+0.107142857142857*cos(2*x(1))-0.0119047619047619*cos(2*x(2))-0.71031746031746);

    dx(3) = (B*B*F*g*sin(2*x(1))/2 - B*C*E*g*sin(x(1))*cos(x(1) - x(2))*cos(x(2)) - B*C*E*g*sin(x(2))*cos(x(1) - x(2))*cos(x(1)) - B*D*F*sin(x(1))*x(4)*x(4) - B*E*E*(sin(x(1) - 2*x(2)) + sin(3*x(1) - 2*x(2)))*x(4)*x(4)/4 + B*E*E*sin(x(1))*cos(x(1) - x(2))*cos(x(1) - x(2))*x(4)*x(4) - B*E*F*sin(x(1) - x(2))*cos(x(1))*x(5)*x(5) + C*C*D*g*sin(2*x(2))/2 + C*D*E*sin(x(1) - x(2))*cos(x(2))*x(4)*x(4) - C*D*F*sin(x(2))*x(5)*x(5) + C*E*E*(sin(2*x(1) - 3*x(2)) + sin(2*x(1) - x(2)))*x(5)*x(5)/4 + C*E*E*sin(x(2))*cos(x(1) - x(2))*cos(x(1) - x(2))*x(5)*x(5) - D*F*u(0) + E*E*u(0)*cos(x(1) - x(2))*cos(x(1) - x(2))) / (-A*D*F + A*E*E*cos(x(1) - x(2))*cos(x(1) - x(2)) + B*B*F*cos(x(1))*cos(x(1)) - 2*B*C*E*cos(x(1) - x(2))*cos(x(1))*cos(x(2)) + C*C*D*cos(x(2))*cos(x(2)));
    dx(4) = (2*A*B*F*g*sin(x(1)) + A*C*E*g*sin(x(1) - 2*x(2)) - A*C*E*g*sin(x(1)) - A*E*E*sin(2*x(1) - 2*x(2))*x(4)*x(4) - 2*A*E*F*sin(x(1) - x(2))*x(5)*x(5) - B*B*F*sin(2*x(1))*x(4)*x(4) - B*C*C*g*sin(x(1) - 2*x(2)) - B*C*C*g*sin(x(1)) + B*C*E*sin(2*x(1) - 2*x(2))*x(4)*x(4) + B*C*E*sin(2*x(1))*x(4)*x(4) + B*C*F*sin(x(1) - x(2))*x(5)*x(5) - B*C*F*sin(x(1) + x(2))*x(5)*x(5) - 2*B*F*u(0)*cos(x(1)) + C*C*E*sin(x(1) - x(2))*x(5)*x(5) + C*C*E*sin(x(1) + x(2))*x(5)*x(5) + C*E*u(0)*cos(x(1) - 2*x(2)) + C*E*u(0)*cos(x(1))) / (2*(A*D*F - A*E*E*cos(x(1) - x(2))*cos(x(1) - x(2)) - B*B*F*cos(x(1))*cos(x(1)) + 2*B*C*E*cos(x(1) - x(2))*cos(x(1))*cos(x(2)) - C*C*D*cos(x(2))*cos(x(2))));
    dx(5) = (-A*B*E*g*sin(2*x(1) - x(2)) - A*B*E*g*sin(x(2)) + 2*A*C*D*g*sin(x(2)) + 2*A*D*E*sin(x(1) - x(2))*x(4)*x(4) + A*E*E*sin(2*x(1) - 2*x(2))*x(5)*x(5) + B*B*C*g*sin(2*x(1) - x(2)) - B*B*C*g*sin(x(2)) - B*B*E*sin(x(1) - x(2))*x(4)*x(4) + B*B*E*sin(x(1) + x(2))*x(4)*x(4) - B*C*D*sin(x(1) - x(2))*x(4)*x(4) - B*C*D*sin(x(1) + x(2))*x(4)*x(4) - B*C*E*sin(2*x(1) - 2*x(2))*x(5)*x(5) + B*C*E*sin(2*x(2))*x(5)*x(5) + B*E*u(0)*cos(2*x(1) - x(2)) + B*E*u(0)*cos(x(2)) - C*C*D*sin(2*x(2))*x(5)*x(5) - 2*C*D*u(0)*cos(x(2))) / (2*(A*D*F - A*E*E*cos(x(1) - x(2))*cos(x(1) - x(2)) - B*B*F*cos(x(1))*cos(x(1)) + 2*B*C*E*cos(x(1) - x(2))*cos(x(1))*cos(x(2)) - C*C*D*cos(x(2))*cos(x(2))));
}

double getEnergyCost(const mpc::cvec<num_states> &x) {
    double E_kin_cart = 0.5 * m0 * pow(x(3), 2);
    double E_kin_p1 = 0.5 * m1 * (pow(x(3) + l1 * x(4) * cos(x(1)), 2) + pow(l1 * x(4) * sin(x(1)), 2)) + 0.5 * I1 * pow(x(4),2);
    double E_kin_p2 = 0.5 * m2 * (pow(x(3) + L1 * x(4) * cos(x(1)) + l2 * x(5) * cos(x(2)), 2) +
                                                                pow(L1 * x(4) * sin(x(1)) + l2 * x(5) * sin(x(2)), 2)) + 0.5 * I2 * pow(x(5), 2);
    double E_kin = E_kin_cart + E_kin_p1 + E_kin_p2;
    double E_pot = m1 * g * l1 * cos(x(1)) + m2 * g * (L1 * cos(x(1)) + l2 * cos(x(2)));
    return (E_kin - E_pot);
}

int main()
{
    Eigen::DiagonalMatrix<double, num_states> Q;
    Q.diagonal() << 0.5, 2.0, 2.0, 0.01, 0.01, 0.01;
    Eigen::DiagonalMatrix<double, num_states> Qf;
    Qf.diagonal() << 60.0, 5.0, 10.0, 0.5, 1.0, 1.0;
    double R = 0.1;
    double weight_stage_energy = 1;
    double weight_terminal_energy = 30;

    mpc::cvec<num_states> xmin, xmax;
    xmax << 6, 10, 10, 30, 30, 30;
    xmin << -6, -10, -10, -30, -30, -30;

    double umax = 20;
    double umin = -umax;

    mpc::NLMPC<
        num_states, num_inputs, num_output,
        pred_hor, ctrl_hor,
        ineq_c, eq_c>
        controller;

    controller.setLoggerLevel(mpc::Logger::LogLevel::NORMAL);
    controller.setDiscretizationSamplingTime(ts);

    mpc::NLParameters params;
    params.maximum_iteration = 3;
    params.relative_ftol = -1;
    params.relative_xtol = -1;
    params.hard_constraints = false;
    params.enable_warm_start = true;

    controller.setOptimizerParameters(params);

    controller.setStateSpaceFunction([&](
                                        mpc::cvec<num_states> &dx,
                                        const mpc::cvec<num_states> &x,
                                        const mpc::cvec<num_inputs> &u,
                                        const unsigned int &)
                                    { doubleInvPendulumDynamics(dx, x, u); });

    auto objEq = [&](
                     const mpc::mat<pred_hor + 1, num_states> &x,
                     const mpc::mat<pred_hor + 1, num_output> &y,
                     const mpc::mat<pred_hor + 1, num_inputs> &u,
                     const double &e)
    {
        double cost = 0;
        for (int i = 0; i < pred_hor; i++)
        {
            cost += x.row(i) * Q * x.row(i).transpose();
            cost += u.row(i) * R * u.row(i).transpose();
            cost += weight_stage_energy * getEnergyCost(x.row(i));
        }
        cost += x.row(pred_hor) * Qf * x.row(pred_hor).transpose();
        cost += u.row(pred_hor) * R * u.row(pred_hor).transpose();
        cost += weight_terminal_energy * getEnergyCost(x.row(pred_hor));
        
        return cost;

        // return x.array().square().sum() + u.array().square().sum();
    };
    controller.setObjectiveFunction(objEq);

    auto conIneq = [&](
                       mpc::cvec<ineq_c> &ineq,
                       const mpc::mat<pred_hor + 1, num_states> &x,
                       const mpc::mat<pred_hor + 1, num_output> &y,
                       const mpc::mat<pred_hor + 1, num_inputs> &u,
                       const double &)
    {
        int idx = 0;
        for (int i = 0; i < pred_hor+1; i++) {
            for (int j = 0; j < num_states; j++)
            {
                ineq(idx++) = x(i, j) - xmax(j);
                ineq(idx++) = xmin(j) - x(i, j);
            }
            ineq(idx++) = u(i, 0) - umax;
            ineq(idx++) = umin - u(i, 0);
        }
    };
    controller.setIneqConFunction(conIneq);

    mpc::cvec<num_states> modelX, modeldX;
    modelX.resize(num_states);
    modeldX.resize(num_states);

    modelX(0) = 0;
    modelX(1) = 0.99*3.14;
    modelX(2) = 0.99*3.14;
    modelX(3) = 0;
    modelX(4) = 0;
    modelX(5) = 0;

    auto r = controller.getLastResult();
    // r.cmd.setZero();
    
    std::ofstream myfile;
    myfile.open("try4.csv");
    double t = 0;
    for(int i = 0; i < 200; i++)
    {
        r = controller.optimize(modelX, r.cmd);
        doubleInvPendulumDynamics(modeldX, modelX, r.cmd);
        modelX += ts * modeldX;
        t += ts;

        myfile << t << "," << modelX(0) << "," << modelX(1) << "," << modelX(2) << "," << modelX(3) << "," << modelX(4) << "," << modelX(5) << "\n";

        std::cout << t << " : time" << ", " << modelX(0) << ", " << modelX(1) << ", " << modelX(2) << ", " << modelX(3) << ", " << modelX(4) << ", " << modelX(5);
        std::cout << std::endl;
        std::cout << "control input is " << r.cmd.transpose() << std::endl;
    }
    myfile.close();

    std::cout << controller.getExecutionStats();

    return 0;
}
