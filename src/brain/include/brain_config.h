#pragma once

#include <Eigen/Core>
#include <ostream>
#include <string>

#include "RoboCupGameControlData.h"
#include "types.h"
#include "utils/math.h"

using namespace std;

/**
 * 存储 Brain 需要的一些配置值，这些值应该是初始化就确认好了，在机器人决策过程中只读不改的
 * 需要在决策过程中变化的值，应该放到 BrainData 中
 * 注意：
 * 1、配置文件会从 config/config.yaml 中读取
 * 2、如果存在 config/config_local.yaml，将会覆盖 config/config.yaml 的值
 *
 */
class BrainConfig {
public:
  int teamId;
  int playerId;
  string fieldType;
  string playerRole;

  double robotHeight;
  double robotOdomFactor;
  double vxFactor;
  double yawOffset;

  bool enableCom;

  bool rerunLogEnableTCP;
  string rerunLogServerIP;
  bool rerunLogEnableFile;
  string rerunLogLogDir;
  double rerunLogMaxFileMins;

  int rerunLogImgInterval;

  string treeFilePath;

  FieldDimensions fieldDimensions;
  vector<FieldLine> mapLines;
  vector<MapMarking> mapMarkings;

  int numOfPlayers = 2;

  double collisionThreshold;
  double safeDistance;
  double avoidSecs;

  double camPixX = 1280;
  double camPixY = 720;

  double camAngleX = deg2rad(90);
  double camAngleY = deg2rad(65);

  double camfx = 643.898;
  double camfy = 643.216;
  double camcx = 649.038;
  double camcy = 357.21;

  Eigen::Matrix4d camToHead;

  double headYawLimitLeft = 1.1;
  double headYawLimitRight = -1.1;
  double headPitchLimitUp = 0.45;
  double vxLimit = 0.8;
  double vyLimit = 0.4;
  double vthetaLimit = 1.0;

  double safeDist = 5.0;
  double goalPostMargin = 0.4;
  double goalPostMarginForTouch = 0.1;
  double ballConfidenceThreshold;
  bool treatPersonAsRobot = false;
  double ballOutThreshold = 2.0;
  double tmBallDistThreshold = 4.0;
  bool limitNearBallSpeed = true;
  double nearBallSpeedLimit = 0.2;
  double nearBallRange = 3.0;

  int numParticles = 150;
  double initFieldMargin = 1.0;
  double alpha1 = 0.1;
  double alpha2 = 0.05;
  double alpha3 = 0.05;
  double alpha4 = 0.01;
  double invNormVar = 1.4;
  double invPerpVar = 4.0;
  double likelihoodWeight = 0.3;
  double unmatchedPenaltyConfThr = 0.6;
  double clusterDistThr = 0.3;
  double clusterThetaThr = 0.35; // ~20 deg
  double clusterMinWeight = 0.05;
  double orientationGatingThr = 1.6;
  double smoothAlpha = 0.4;
  double essThreshold = 0.4;

  // DWA Parameters
  double dwaMaxVelX = 0.5;
  double dwaMinVelX = -0.2;
  double dwaMaxVelTheta = 2.0;
  double dwaAccLimX = 2.0;
  double dwaAccLimTheta = 3.0;
  double dwaAccLimY = 1.0;
  double dwaWGoal = 1.0;
  double dwaWAlign = 0.8;
  double dwaWObs = 2.0;
  double dwaSafeDist = 0.5;

  bool soundEnable = false;
  string soundPack = "espeak";

  void calcMapLines();
  void calcMapMarkings();

  void handle();

  void print(ostream &os);
};