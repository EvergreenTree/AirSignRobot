// Supervised finite clockwise turn using independent scan registration.
// Fresh camera inspection and a locally held physical stop are required.
// Wheel odometry is logged, not used as independent chassis pose during steering.
#include <franka/robot.h>
#include <franka/control_types.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
volatile std::sig_atomic_t interrupted=0;
void on_signal(int){interrupted=1;}
double clock_s(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Sample{double t,command,x,y,yaw,now,sensor_stamp,monitor_stamp;long sequence;bool valid;int reason;franka::RobotState state;};
struct MonitorSample{double started,finished,sensor,written;long sequence;bool valid;int error;};
std::array<Sample,1100> samples{};
std::array<MonitorSample,3000> monitor_samples{};
int main(int argc,char**argv){
  const bool observe_only=argc==2&&std::string(argv[1])=="--observe-only";
  const bool zero_only=argc==2&&std::string(argv[1])=="--supervised-zero-check";
  double target=0;
  if(!observe_only&&!zero_only){
    if(argc!=3||std::string(argv[1])!="--supervised-right-turn")return 2;
    char* end=nullptr;target=std::strtod(argv[2],&end);
    if(end==argv[2]||*end||!std::isfinite(target)||target<-.15||target>-.02)return 2;
  }
  std::signal(SIGINT,on_signal);std::signal(SIGTERM,on_signal);std::signal(SIGHUP,on_signal);
  std::atomic<bool> finish{false},valid{false};
  std::atomic<double> stamp{0},reference{0},x{0},y{0},angle{0},monitor_stamp{0};
  std::atomic<long> sequence{-1};
  unsigned nm=0;
  std::thread monitor([&]{
    while(!finish){
      MonitorSample trace{clock_s(),0,0,0,-1,false,0};
      try{
        std::ifstream file("/tmp/airsign-lidar-motion.json");nlohmann::json j;file>>j;
        bool good=j.at("valid").get<bool>();
        trace.sequence=j.at("sequence").get<long>();
        trace.written=j.at("written_monotonic").get<double>();
        if(good){
          const double a=j.at("x"),b=j.at("y"),c=j.at("yaw");
          const double sensor=j.at("sensor_monotonic"),written=j.at("written_monotonic");
          trace.sensor=sensor;
          const double compute=j.at("compute_seconds").get<double>();
          // End-to-end sensor age is enforced in the control callback. A
          // separate 40 ms calculation cutoff belonged to the former 25 Hz
          // loop and incorrectly rejected fresh, accurate results after the
          // observer changed to consume each 20 Hz relay update promptly.
          good=std::isfinite(a)&&std::isfinite(b)&&std::isfinite(c)&&
               j.at("inlier_ratio").get<double>()>=.8&&j.at("median_residual").get<double>()<=.009&&
               std::isfinite(sensor)&&std::isfinite(written)&&sensor<=written&&written<=clock_s()&&
               std::isfinite(compute)&&compute>=0;
          x=a;y=b;angle=c;stamp=std::min(sensor,written);reference=j.at("reference_monotonic").get<double>();
        }
        valid=good;trace.valid=good;
        sequence=trace.sequence;
      }catch(...){valid=false;trace.error=1;}
      trace.finished=clock_s();monitor_stamp=trace.finished;
      if(nm<monitor_samples.size())monitor_samples[nm++]=trace;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
  unsigned ns=0;int reason=0,result=0;bool reached=false,timed_out=false;
  double elapsed=0,max_dt=0,min_success=1,max_shift=0,max_angle=0,progress=0;
  double abort_age=0,abort_monitor_age=0,abort_time=0;long abort_sequence=-1;bool abort_valid=false;
  try{
    if(observe_only){
      const double until=clock_s()+8;
      while(clock_s()<until&&!interrupted)std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }else{
    const double ready_deadline=clock_s()+1.5;
    while(clock_s()<ready_deadline){
      if(valid&&clock_s()-stamp.load()<=.10&&clock_s()-reference.load()<=10&&
         std::hypot(x.load(),y.load())<=.007&&std::abs(angle.load())<=.005)break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if(!valid||clock_s()-stamp.load()>.12||clock_s()-reference.load()>10||
       std::hypot(x.load(),y.load())>.007||std::abs(angle.load())>.005){
      std::cerr<<"Scene preflight: valid="<<valid.load()<<", age="<<clock_s()-stamp.load()
               <<", reference_age="<<clock_s()-reference.load()<<", xy="<<x.load()<<','<<y.load()
               <<", yaw="<<angle.load()<<'\n';
      throw std::runtime_error("Fresh stationary scene reference required");
    }
    const double sx=x.load(),sy=y.load(),sa=angle.load();
    franka::Robot robot("172.16.16.10",franka::RealtimeConfig::kEnforce);
    const auto before=robot.readOnce();
    if(before.robot_mode!=franka::RobotMode::kIdle||static_cast<bool>(before.current_errors))
      throw std::runtime_error("Requires error-free Idle; no recovery");
    const double start=clock_s();unsigned count=0;double velocity=0,settled=-1;
    robot.control([&](const franka::RobotState& state,franka::Duration duration)->franka::CartesianVelocities{
      const double dt=duration.toSec();elapsed+=dt;max_dt=std::max(max_dt,dt);
      const double px=x.load()-sx,py=y.load()-sy,pa=angle.load()-sa;
      progress=pa;max_shift=std::max(max_shift,std::hypot(px,py));max_angle=std::max(max_angle,std::abs(pa));
      if(elapsed>.5)min_success=std::min(min_success,state.control_command_success_rate);
      if(!reason){
        if(interrupted)reason=1;
        else if(!valid||clock_s()-stamp.load()>.16)reason=2;
        else if(std::hypot(px,py)>(zero_only?.012:.02)||std::abs(pa)>(zero_only?.015:std::abs(target)+.02)||pa>.008)reason=3;
        else if(dt>.01||clock_s()-start>10)reason=4;
        if(reason){abort_time=clock_s();abort_age=abort_time-stamp.load();
          abort_monitor_age=abort_time-monitor_stamp.load();abort_sequence=sequence.load();abort_valid=valid.load();}
      }
      if(elapsed>=8)timed_out=true;
      const double error=target-progress;
      if(!zero_only&&elapsed>.5&&(std::abs(error)<.003||error>0))reached=true;
      if(zero_only&&elapsed>=6)reached=true;
      double desired=0;
      if(!zero_only&&elapsed>.5&&!reason&&!timed_out&&!reached)desired=std::clamp(2*error,-.05,0.0);
      velocity+=std::clamp(desired-velocity,-.12*dt,.12*dt);
      if(count++%10==0&&ns<samples.size())samples[ns++]={elapsed,velocity,px,py,pa,clock_s(),stamp.load(),monitor_stamp.load(),sequence.load(),valid.load(),reason,state};
      franka::CartesianVelocities command({0,0,0,0,0,velocity});
      if((reason||timed_out||reached)&&std::abs(velocity)<1e-8){
        if(settled<0)settled=elapsed;
        if(elapsed-settled>=.5)return franka::MotionFinished(command);
      }
      return command;
    },franka::ControllerMode::kJointImpedance,true);
    const auto after=robot.readOnce();
    std::cout<<std::setprecision(12)<<"{\"target_rad\":"<<target<<",\"elapsed_s\":"<<elapsed
      <<",\"abort_reason\":"<<reason<<",\"scene_yaw_rad\":"<<progress<<",\"max_scene_translation_m\":"<<max_shift
      <<",\"max_scene_yaw_rad\":"<<max_angle<<",\"max_dt_s\":"<<max_dt
      <<",\"target_reached\":"<<(reached?"true":"false")<<",\"timed_out\":"<<(timed_out?"true":"false")
      <<",\"min_command_success\":"<<min_success<<",\"final_mode\":"<<static_cast<int>(after.robot_mode)
      <<",\"abort_time\":"<<abort_time<<",\"abort_sensor_age\":"<<abort_age<<",\"abort_monitor_age\":"<<abort_monitor_age
      <<",\"abort_sequence\":"<<abort_sequence<<",\"abort_valid\":"<<(abort_valid?"true":"false")
      <<",\"final_errors\":"<<std::quoted(static_cast<std::string>(after.current_errors))<<"}\n";
    if(reason||timed_out||static_cast<bool>(after.current_errors))result=4;
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
  finish=true;monitor.join();
  std::ofstream output("/tmp/airsign-scene-turn-feedback.jsonl");output<<std::setprecision(12);
  for(unsigned i=0;i<ns;++i){auto&s=samples[i];output<<"{\"t\":"<<s.t<<",\"command\":"<<s.command<<",\"scene_xyyaw\":["<<s.x<<','<<s.y<<','<<s.yaw<<"],\"now\":"<<s.now<<",\"sensor_stamp\":"<<s.sensor_stamp<<",\"monitor_stamp\":"<<s.monitor_stamp<<",\"sequence\":"<<s.sequence<<",\"valid\":"<<(s.valid?"true":"false")<<",\"reason\":"<<s.reason<<",\"state\":"<<s.state<<"}\n";}
  std::ofstream mout("/tmp/airsign-scene-monitor.jsonl");mout<<std::setprecision(15);
  for(unsigned i=0;i<nm;++i){auto&s=monitor_samples[i];mout<<"{\"started\":"<<s.started<<",\"finished\":"<<s.finished<<",\"sensor\":"<<s.sensor<<",\"written\":"<<s.written<<",\"sequence\":"<<s.sequence<<",\"valid\":"<<(s.valid?"true":"false")<<",\"error\":"<<s.error<<"}\n";}
  return result;
}
