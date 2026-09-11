// Read-only state capture. No control, recovery, or configuration calls.
#include <franka/robot.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    std::ofstream out(argv[1]);
    if (!out) return 2;
    out << std::setprecision(12);
    franka::Robot robot("172.16.16.10", franka::RealtimeConfig::kIgnore);
    auto start=std::chrono::steady_clock::now();
    unsigned count=0;
    robot.read([&](const franka::RobotState& state) {
      if (count++ % 20 == 0) out << state << '\n';
      return std::chrono::steady_clock::now()-start<std::chrono::seconds(3);
    });
    std::cout << "{\"motion_commands_sent\":false,\"read_count\":" << count << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
