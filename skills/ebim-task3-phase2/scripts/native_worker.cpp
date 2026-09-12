// Persistent process/connection, finite pulses. Protocol I/O never runs in the servo callback.
// The default build is a hardware-free timing mock. The live build is experimental.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#ifdef AIRSIGN_LIVE
#include <franka/robot.h>
#include <franka/control_types.h>
#endif

using Clock = std::chrono::steady_clock;
static std::int64_t ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
static volatile std::sig_atomic_t interrupted = 0;
static void signal_stop(int) { interrupted = 1; }
struct State { std::array<double,16> pose{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}; };
struct Command { std::string op; std::uint64_t id=0, anchor=0, generation=0; std::int64_t deadline=0; int axis=0; double distance=0, seconds=0; };
std::mutex output_mutex, mailbox_mutex;
std::condition_variable wake;
std::optional<Command> mailbox;
std::atomic<bool> closing{false}, busy{false};
std::atomic<std::uint64_t> generation{0};

void emit(const std::string& event, std::uint64_t id, const std::string& fields="") {
  std::lock_guard<std::mutex> lock(output_mutex);
  std::cout << "{\"event\":\"" << event << "\",\"id\":" << id << ",\"clock_ns\":" << ns() << fields << "}" << std::endl;
}
std::string pose_json(const State& s) {
  std::ostringstream out; out << std::setprecision(12) << ",\"pose\":[";
  for(size_t i=0;i<16;i++) out << (i?",":"") << s.pose[i];
  out << ']'; return out.str();
}
void reader() {
  std::string buffer;
  std::uint64_t previous=0;
  while(!closing && !interrupted) {
    pollfd fd{STDIN_FILENO, POLLIN, 0};
    int ready=poll(&fd,1,100);
    if(ready<0) continue;
    if(!ready) continue;
    char chunk[256]; auto count=::read(STDIN_FILENO,chunk,sizeof(chunk));
    if(count<=0) break;
    buffer.append(chunk,static_cast<size_t>(count));
    size_t end;
    while((end=buffer.find('\n'))!=std::string::npos) {
      std::string line=buffer.substr(0,end); buffer.erase(0,end+1);
      Command c; std::istringstream in(line); std::string extra;
      if(!(in>>c.op>>c.id) || c.id==0 || c.id<=previous) { emit("rejected",c.id,",\"reason\":\"sequence\""); continue; }
      previous=c.id;
      if(c.op=="MOVE") {
        if(!(in>>c.anchor>>c.deadline>>c.axis>>c.distance>>c.seconds) || (in>>extra) ||
           c.axis<0 || c.axis>2 || !std::isfinite(c.distance) || std::abs(c.distance)>.005 ||
           std::abs(c.distance)<.0001 || !std::isfinite(c.seconds) || c.seconds<.6 || c.seconds>1.0 ||
           c.deadline<=ns() || c.deadline-ns()>5000000000LL) {
          emit("rejected",c.id,",\"reason\":\"bounds_or_expired\""); continue;
        }
      } else if(in>>extra) { emit("rejected",c.id,",\"reason\":\"syntax\""); continue; }
      if(c.op=="PING") { emit("pong",c.id); continue; }
      if(c.op=="STOP" || c.op=="QUIT") {
        ++generation;
        if(c.op=="QUIT") closing=true;
        emit("stop_requested",c.id); wake.notify_all(); continue;
      }
      if(c.op!="MOVE" && c.op!="READ") { emit("rejected",c.id,",\"reason\":\"unknown_command\""); continue; }
      bool expected=false;
      if(!busy.compare_exchange_strong(expected,true)) { emit("rejected",c.id,",\"reason\":\"busy_no_queue\""); continue; }
      c.generation=generation.load();
      { std::lock_guard<std::mutex> lock(mailbox_mutex); mailbox=c; }
      wake.notify_one();
    }
    if(buffer.size()>256) { emit("protocol_error",0); break; }
  }
  closing=true; ++generation; wake.notify_all();
}

class Backend {
  State mock;
#ifdef AIRSIGN_LIVE
  franka::Robot robot;
#endif
public:
  explicit Backend(const std::string& ip)
#ifdef AIRSIGN_LIVE
    : robot(ip, franka::RealtimeConfig::kEnforce)
#endif
    { (void)ip; }
  State read() {
#ifdef AIRSIGN_LIVE
    auto s=robot.readOnce();
    if(s.robot_mode!=franka::RobotMode::kIdle || s.current_errors) throw std::runtime_error("Requires error-free Idle");
    for(double v:s.dq) if(std::abs(v)>.02) throw std::runtime_error("Already moving");
    return {s.O_T_EE};
#else
    return mock;
#endif
  }
  bool move(const Command& c, const State& expected) {
    bool stopped=false;
#ifdef AIRSIGN_LIVE
    auto before=robot.readOnce();
    if(before.robot_mode!=franka::RobotMode::kIdle || before.current_errors) throw std::runtime_error("Not ready");
    for(double v:before.dq) if(std::abs(v)>.02) throw std::runtime_error("Already moving");
    for(size_t i=0;i<16;i++) if(std::abs(before.O_T_EE[i]-expected.pose[i])>.003) throw std::runtime_error("Pose changed since frame");
    double elapsed=0; size_t callbacks=0; const auto wall=ns();
    robot.control([&](const franka::RobotState& s, franka::Duration period)->franka::CartesianVelocities {
      elapsed+=period.toSec(); ++callbacks;
      bool guard=interrupted || closing || generation!=c.generation || s.current_errors ||
                 (callbacks>2 && period.toSec()>.01) || ns()-wall>2000000000LL;
      // Check expiry again in the first servo callback: transport/setup may have taken time.
      if(callbacks==1 && ns()>=c.deadline) guard=true;
      double travel=0;
      for(size_t i=0;i<3;i++) travel+=std::pow(s.O_T_EE[12+i]-before.O_T_EE[12+i],2);
      if(std::sqrt(travel)>std::abs(c.distance)+.003) guard=true;
      for(size_t i=0;i<12;i++) if(std::abs(s.O_T_EE[i]-before.O_T_EE[i])>.02) guard=true;
      for(size_t i=0;i<7;i++) if(s.joint_collision[i] || std::abs(s.dq[i])>.2 || std::abs(s.q[i]-before.q[i])>.08) guard=true;
      for(size_t i=0;i<6;i++) if(s.cartesian_collision[i] || std::abs(s.O_F_ext_hat_K[i]-before.O_F_ext_hat_K[i])>(i<3?8.:2.)) guard=true;
      if(guard) { stopped=true; return franka::MotionFinished(franka::CartesianVelocities({0,0,0,0,0,0})); }
      if(elapsed>=c.seconds) return franka::MotionFinished(franka::CartesianVelocities({0,0,0,0,0,0}));
      double u=elapsed/c.seconds;
      std::array<double,6> velocity{}; velocity[c.axis]=c.distance*30*u*u*(1-u)*(1-u)/c.seconds;
      return franka::CartesianVelocities(velocity);
    },franka::ControllerMode::kCartesianImpedance,true);
#else
    (void)expected;
    const auto start=ns(); double previous=0;
    while(true) {
      if(interrupted || closing || generation!=c.generation) { stopped=true; break; }
      double u=std::min(1., (ns()-start)*1e-9/c.seconds);
      double position=10*std::pow(u,3)-15*std::pow(u,4)+6*std::pow(u,5);
      mock.pose[12+c.axis]+=c.distance*(position-previous); previous=position;
      if(u>=1.) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
    return stopped;
  }
};

int main(int argc,char**argv) {
  std::signal(SIGINT,signal_stop); std::signal(SIGTERM,signal_stop); std::signal(SIGHUP,signal_stop);
  std::string ip;
#ifdef AIRSIGN_LIVE
  if(argc!=4 || std::string(argv[1])!="--execute" || std::string(argv[2])!="--ip") { std::cerr<<"Requires --execute --ip ARM_IP and an attended, normally initialized FCI session\n"; return 2; }
  ip=argv[3]; const std::string mode="franka";
#else
  if(argc!=2 || std::string(argv[1])!="--mock") { std::cerr<<"usage: gamepad_mock --mock\n"; return 2; }
  const std::string mode="mock";
#endif
  try {
    Backend backend(ip); backend.read();
    emit("ready",0,",\"backend\":\""+mode+"\"");
    std::thread input(reader);
    std::uint64_t anchor_id=0, anchor_generation=0; std::int64_t anchor_time=0; State anchor;
    int result=0;
    while(!closing && !interrupted) {
      Command c;
      { std::unique_lock<std::mutex> lock(mailbox_mutex);
        if(!wake.wait_for(lock,std::chrono::seconds(30),[]{return mailbox.has_value() || closing || interrupted;})) { closing=true; break; }
        if(closing || interrupted) break;
        c=*mailbox; mailbox.reset();
      }
      std::string event="rejected", fields=",\"reason\":\"cancelled\"";
      try {
        if(c.generation!=generation || closing) {}
        else if(c.op=="READ") {
          anchor=backend.read(); anchor_id=c.id; anchor_time=ns(); anchor_generation=c.generation;
          event="state"; fields=pose_json(anchor);
        } else if(c.anchor!=anchor_id || anchor_id==0 || c.generation!=anchor_generation || ns()-anchor_time>5000000000LL || ns()>=c.deadline) {
          fields=",\"reason\":\"anchor_or_deadline\"";
        } else {
          anchor_id=0; const auto start=ns();
          emit("started",c.id);
          bool stopped=backend.move(c,anchor);
          State after=backend.read();
          event="result"; fields=",\"status\":\""+std::string(stopped?"stopped":"complete")+"\",\"pulse_ms\":"+std::to_string((ns()-start)/1e6)+pose_json(after);
          if(stopped) { result=3; closing=true; }
        }
      } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n'; event="fault"; fields=",\"reason\":\"backend_refused_or_failed\""; result=1; closing=true;
      }
      busy=false;
      emit(event,c.id,fields);
    }
    closing=true; ++generation; input.join(); emit("closed",0); return result;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
