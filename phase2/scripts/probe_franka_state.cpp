// Read measured state only. No control(), recovery, configuration, or motion calls.
#include <franka/robot.h>
#include <array>
#include <exception>
#include <iomanip>
#include <iostream>

template <std::size_t N>
void print_array(const std::array<double, N>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: probe_franka_state ROBOT_IP\n";
    return 2;
  }
  try {
    // This read-only process does not run a real-time control loop.
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const auto state = robot.readOnce();
    std::cout << std::setprecision(12) << std::boolalpha;
    std::cout << "{\"motion_commands_sent\":false,\"robot_mode_code\":"
              << static_cast<int>(state.robot_mode)
              << ",\"has_current_errors\":" << static_cast<bool>(state.current_errors)
              << ",\"current_errors\":" << std::quoted(static_cast<std::string>(state.current_errors))
              << ",\"last_motion_errors\":" << std::quoted(static_cast<std::string>(state.last_motion_errors))
              << ",\"q\":";
    print_array(state.q);
    std::cout << ",\"dq\":";
    print_array(state.dq);
    std::cout << ",\"tau_ext_hat_filtered\":";
    print_array(state.tau_ext_hat_filtered);
    std::cout << ",\"O_F_ext_hat_K\":";
    print_array(state.O_F_ext_hat_K);
    std::cout << ",\"O_T_EE\":";
    print_array(state.O_T_EE);
    std::cout << ",\"F_T_EE\":";
    print_array(state.F_T_EE);
    std::cout << ",\"F_T_NE\":";
    print_array(state.F_T_NE);
    std::cout << ",\"NE_T_EE\":";
    print_array(state.NE_T_EE);
    std::cout << ",\"EE_T_K\":";
    print_array(state.EE_T_K);
    std::cout << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "Read-only arm probe failed: " << error.what() << '\n';
    return 1;
  }
}
