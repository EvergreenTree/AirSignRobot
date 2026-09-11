// Bounded, measured-pose-anchored translation. Never homes, rotates, or recovers faults.
#include <franka/robot.h>
#include <franka/control_types.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }
double norm3(const std::array<double, 3>& a) {
  return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}
int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: native_arm_translation IP dx dy dz seconds expected_pose_file --execute\n";
    return 2;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  try {
    std::array<double,3> delta{std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4])};
    double duration = std::stod(argv[5]);
    if (!std::isfinite(norm3(delta)) || norm3(delta) > .050001 || norm3(delta) < .0001 ||
        !std::isfinite(duration) || duration < 3 || duration > 8 || std::string(argv[7]) != "--execute")
      throw std::runtime_error("Rejected translation bounds");
    std::array<double,16> expected{};
    std::ifstream pose_file(argv[6]);
    for (auto& v : expected) if (!(pose_file >> v) || !std::isfinite(v))
      throw std::runtime_error("Invalid expected measured pose file");
    franka::Robot robot(argv[1], franka::RealtimeConfig::kEnforce);
    auto before = robot.readOnce();
    if (before.robot_mode != franka::RobotMode::kIdle || before.current_errors)
      throw std::runtime_error("Robot is not idle and error-free");
    for (size_t i=0;i<16;i++)
      if (std::abs(before.O_T_EE[i]-expected[i]) > .003)
        throw std::runtime_error("Measured pose changed since review");
    for(double v: before.dq) if(std::abs(v)>.02)
      throw std::runtime_error("Arm is already moving");
    std::cout << std::setprecision(12) << "Starting bounded translation; duration=" << duration << std::endl;
    const auto initial = before.O_T_EE;
    const auto initial_force = before.O_F_ext_hat_K;
    const auto wall_start = std::chrono::steady_clock::now();
    double t=0;
    size_t callbacks=0;
    bool guard=false;
    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianVelocities {
      ++callbacks;
      t += period.toSec();
      double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count();
      std::array<double,3> travelled{};
      for(size_t i=0;i<3;i++) travelled[i]=state.O_T_EE[12+i]-initial[12+i];
      bool contact=false;
      for(size_t i=0;i<6;i++) {
        double limit=i<3?8.0:2.0;
        if(std::abs(state.O_F_ext_hat_K[i]-initial_force[i])>limit) contact=true;
        if(state.cartesian_collision[i]) contact=true;
      }
      for(size_t i=0;i<7;i++) {
        if(state.joint_collision[i] || std::abs(state.dq[i])>.2 || std::abs(state.q[i]-before.q[i])>.16) contact=true;
      }
      bool rotation=false;
      for(size_t i=0;i<12;i++) if(std::abs(state.O_T_EE[i]-initial[i])>.02) rotation=true;
      if(interrupted || contact || rotation || state.current_errors ||
         norm3(travelled)>norm3(delta)+.008 || wall>duration+3 || (callbacks>2 && period.toSec()>.01)) {
        guard=true;
        return franka::MotionFinished(franka::CartesianVelocities({0,0,0,0,0,0}));
      }
      if(t>=duration) return franka::MotionFinished(franka::CartesianVelocities({0,0,0,0,0,0}));
      double u=t/duration;
      double scale=30*u*u*(1-u)*(1-u)/duration;
      return franka::CartesianVelocities({delta[0]*scale,delta[1]*scale,delta[2]*scale,0,0,0});
    },franka::ControllerMode::kCartesianImpedance,true);
    auto after=robot.readOnce();
    std::cout << "{\"guard_stopped\":" << (guard?"true":"false") << ",\"callbacks\":" << callbacks << ",\"delta_measured\":[";
    for(size_t i=0;i<3;i++)std::cout << (i?",":"") << after.O_T_EE[12+i]-initial[12+i];
    std::cout << "],\"errors\":\"" << static_cast<std::string>(after.current_errors) << "\"}" << std::endl;
    return guard?3:0;
  } catch(const std::exception& e) {
    std::cerr << "Arm translation stopped/refused: " << e.what() << std::endl;
    return 1;
  }
}
