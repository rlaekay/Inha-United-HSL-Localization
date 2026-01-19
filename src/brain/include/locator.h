
#pragma once

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <behaviortree_cpp/bt_factory.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <rerun.hpp>

#include "types.h"

#include <map>
#include <string>

using namespace std;
namespace chr = std::chrono;

class Locator {
public:
  double convergeTolerance = 0.2;
  double residualTolerance = 0.4;
  double maxIteration = 20;
  double muOffset = 2.0;
  double numShrinkRatio = 0.85;
  double offsetShrinkRatio = 0.8;
  int minMarkerCnt = 3;
  double enableLog = true;
  string logIP = "127.0.0.1:9876";

  rerun::RecordingStream log = rerun::RecordingStream("locator", "locator");
  vector<FieldMarker> fieldMarkers;
  // Pre-categorized markers
  map<char, vector<FieldMarker>> mapByType;

  FieldDimensions fieldDimensions;
  Eigen::ArrayXXd hypos;
  PoseBox2D constraints;
  double offsetX, offsetY, offsetTheta;
  Pose2D bestPose;
  double bestResidual;

  void init(FieldDimensions fd, int minMarkerCnt = 4, double residualTolerance = 0.4, double muOffsetParam = 2.0, bool enableLog = false,
            string logIP = "127.0.0.1:9876");

  void calcFieldMarkers(FieldDimensions fd);

  LocateResult locateRobot(vector<FieldMarker> markers_r, PoseBox2D constraints, int numParticles = 200, double offsetX = 2.0, double offsetY = 2.0,
                           double offsetTheta = M_PI / 4);

  int genInitialParticles(int num = 200);

  int genParticles();

  FieldMarker markerToFieldFrame(FieldMarker marker, Pose2D pose);

  double minDist(FieldMarker marker);

  vector<double> getOffset(FieldMarker marker);

  double residual(vector<FieldMarker> markers_r, Pose2D pose);

  bool isConverged();

  int calcProbs(vector<FieldMarker> markers_r);

  Pose2D finalAdjust(vector<FieldMarker> markers_r, Pose2D pose);

  inline double probDesity(double r, double mu, double sigma) {
    if (sigma < 1e-5) return 0.0;
    return 1 / sqrt(2 * M_PI * sigma * sigma) * exp(-(r - mu) * (r - mu) / (2 * sigma * sigma));
  };

  void logParticles();
  void logParticles(double);

  rerun::RecordingStream *logger = nullptr;
  void setLog(rerun::RecordingStream *stream);

  // Main PF Parameters (Reduced)
  int pfNumParticles;
  double pfInitFieldMargin; // Margin around field for random init
  double pfInitGlobalDist;  // (Optional) Max distance for global init

  // Weights need to be carefully managed in MatrixXd context
  // The matrix itself (4 x N) contains x, y, theta, weight
  Eigen::MatrixXd pfParticles;
  double avgWeight;
  double totalWeight;

  // -- Noise Parameters --
  double initSigmaX, initSigmaY, initSigmaTheta;
  double updateSigmaX, updateSigmaY, updateSigmaTheta;
  double sigma; // likelihood sigma

  // -- Movement Thresholds --
  double distThreshold;
  double angleThreshold;

  // -- Flags --
  bool isPFInitialized;
  std::mutex pfMutex;

  // -- Random Engine --
  std::default_random_engine generator;
  Pose2D lastPFOdomPose = {0, 0, 0};
  double pfSensorNoiseR = 1.0;

  // MCL Methods (PF Suffix)
  void globalInitPF(Pose2D currentOdom);
  void predictPF(Pose2D currentOdomPose);
  void correctPF(const vector<FieldMarker> markers);
  Pose2D getEstimatePF();
  bool getIsPFInitialized() const { return isPFInitialized; }

  // Augmented MCL State
  double w_slow = 0.0;
  double w_fast = 0.0;
  double alpha_slow = 0.05;
  double alpha_fast = 0.5;
  double pfInjectionRatio = 0.2;

  int pfNumParticles = 150;
  double pfInitFieldMargin = 1.0;
  bool pfInitOwnHalfOnly = true;

  double pfAlpha1 = 0.08;  // rot -> rot
  double pfAlpha2 = 0.005; // rot -> trans
  double pfAlpha3 = 0.005; // trans -> rot
  double pfAlpha4 = 0.005; // trans -> trans

  double pfZeroMotionTransThresh = 0.01;
  double pfZeroMotionRotThresh = 0.03;
  bool pfResampleWhenStopped = false;
  bool isRobotMoving = true;

  double pfClusterDistThr = 0.3;
  double pfClusterThetaThr = 0.35; // ~20 deg
  double pfSmoothAlpha = 0.4;

  // KLD State
  double kldErr = 0.05;
  double kldZ = 2.33;
  int minParticles = 50;
  int maxParticles = 500;
  double pfResolutionX = 0.2;
  double pfResolutionY = 0.2;
  double pfResolutionTheta = 10.0 * M_PI / 180.0;

  void setPFParams(int numParticles, double initMargin, bool ownHalf, double sensorNoise, std::vector<double> alphas, double alphaSlow, double alphaFast,
                   double injectionRatio, double zeroMotionTransThresh = 0.001, double zeroMotionRotThresh = 0.002, bool resampleWhenStopped = false,
                   double clusterDistThr = 0.3, double clusterThetaThr = 0.35, double smoothAlpha = 0.4, double kldErr = 0.05, double kldZ = 2.33,
                   int minParticles = 50, int maxParticles = 500, double resX = 0.2, double resY = 0.2, double resTheta = 0.17);

  // Pose Smoothing
  Pose2D smoothedPose = {0, 0, 0};
  bool hasSmoothedPose = false;
};

class Brain;
using namespace BT;

void RegisterLocatorNodes(BT::BehaviorTreeFactory &factory, Brain *brain);

class SelfLocate : public SyncActionNode {
public:
  SelfLocate(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<string>("mode", "enter_field", "must be one of [trust_direction, face_forward, fall_recovery]"),
        InputPort<double>("msecs_interval", 10000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
    };
  };

private:
  Brain *brain;
};

class SelfLocateEnterField : public SyncActionNode {
public:
  SelfLocateEnterField(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
    };
  };

private:
  Brain *brain;
};

class SelfLocate1M : public SyncActionNode {
public:
  SelfLocate1M(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
        InputPort<double>("max_dist", 2.0, "marker 距离机器人的距离小于此值时, 才进行校准. (距离小测距更准)"),
        InputPort<double>("max_drift", 1.0, "校准后的位置与原位置距离应小于此值, 否则认为校准失败"),
        InputPort<bool>("validate", true, "校准后, 用其它的 marker 进行验证, 要求小于 locator 的 max residual"),
    };
  };

private:
  Brain *brain;
};

class SelfLocate2X : public SyncActionNode {
public:
  SelfLocate2X(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
        InputPort<double>("max_dist", 2.0, "penalty point 距离机器人的距离小于此值时, 才进行校准. (距离小测距更准)"),
        InputPort<double>("max_drift", 1.0, "校准后的位置与原位置距离应小于此值, 否则认为校准失败"),
        InputPort<bool>("validate", true, "校准后, 用其它的 marker 进行验证, 要求小于 locator 的 max residual"),
    };
  };

private:
  Brain *brain;
};

class SelfLocate2T : public SyncActionNode {
public:
  SelfLocate2T(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
        InputPort<double>("max_dist", 2.0, "两个 TCross 距离机器人的距离小于此值时, 才进行校准. (距离小测距更准)"),
        InputPort<double>("max_drift", 1.0, "校准后的位置与原位置距离应小于此值, 否则认为校准失败"),
        InputPort<bool>("validate", true, "校准后, 用其它的 marker 进行验证, 要求小于 locator 的 max residual"),
    };
  };

private:
  Brain *brain;
};

class SelfLocateLT : public SyncActionNode {
public:
  SelfLocateLT(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
        InputPort<double>("max_dist", 2.0, "penalty point 距离机器人的距离小于此值时, 才进行校准. (距离小测距更准)"),
        InputPort<double>("max_drift", 1.0, "校准后的位置与原位置距离应小于此值, 否则认为校准失败"),
        InputPort<bool>("validate", true, "校准后, 用其它的 marker 进行验证, 要求小于 locator 的 max residual"),
    };
  };

private:
  Brain *brain;
};

class SelfLocatePT : public SyncActionNode {
public:
  SelfLocatePT(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
        InputPort<double>("max_dist", 2.0, "penalty point 距离机器人的距离小于此值时, 才进行校准. (距离小测距更准)"),
        InputPort<double>("max_drift", 1.0, "校准后的位置与原位置距离应小于此值, 否则认为校准失败"),
        InputPort<bool>("validate", true, "校准后, 用其它的 marker 进行验证, 要求小于 locator 的 max residual"),
    };
  };

private:
  Brain *brain;
};

class SelfLocateBorder : public SyncActionNode {
public:
  SelfLocateBorder(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

  NodeStatus tick() override;

  static PortsList providedPorts() {
    return {
        InputPort<double>("msecs_interval", 1000, "防止过于频繁地校准, 如果上一次校准距离现在小于这个时间, 则不重新校准."),
        InputPort<double>("max_dist", 2.0, "border 距离机器人的距离小于此值时, 才进行校准. (距离小测距更准)"),
        InputPort<double>("max_drift", 1.0, "校准后的位置与原位置距离应小于此值, 否则认为校准失败"),
        InputPort<bool>("validate", true, "校准后, 用其它的 marker 进行验证, 要求小于 locator 的 max residual"),
    };
  };

private:
  Brain *brain;
};
