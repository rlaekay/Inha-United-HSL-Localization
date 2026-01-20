#include "locator.h"
#include "brain.h"
#include "brain_tree.h"
#include "utils/hungarian.h"
#include "utils/math.h"
#include "utils/misc.h"
#include "utils/print.h"
#include <map>
#include <random>
#include <set>

#define REGISTER_LOCATOR_BUILDER(Name)                                                                                                                         \
  factory.registerBuilder<Name>(#Name, [brain](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });

// Helper for random number generation
std::mt19937 &getRandomEngine() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return gen;
}

double gaussianRandom(double mean, double stddev) {
  std::normal_distribution<double> d(mean, stddev);
  return d(getRandomEngine());
}

double uniformRandom(double min, double max) {
  std::uniform_real_distribution<double> d(min, max);
  return d(getRandomEngine());
}
void RegisterLocatorNodes(BT::BehaviorTreeFactory &factory, Brain *brain) {
  REGISTER_LOCATOR_BUILDER(SelfLocate);
  REGISTER_LOCATOR_BUILDER(SelfLocateEnterField);
  REGISTER_LOCATOR_BUILDER(SelfLocate1M);
  REGISTER_LOCATOR_BUILDER(SelfLocateBorder);
  REGISTER_LOCATOR_BUILDER(SelfLocate2T);
  REGISTER_LOCATOR_BUILDER(SelfLocateLT);
  REGISTER_LOCATOR_BUILDER(SelfLocatePT);
  REGISTER_LOCATOR_BUILDER(SelfLocate2X);
}

void Locator::calcFieldMarkers(FieldDimensions fd) {

  fieldMarkers.push_back(FieldMarker{'X', 0.0, 0.0, 0.0});

  fieldMarkers.push_back(FieldMarker{'G', fd.length / 2, -fd.goalWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'G', fd.length / 2, fd.goalWidth / 2, 0.0});

  fieldMarkers.push_back(FieldMarker{'G', -fd.length / 2, -fd.goalWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'G', -fd.length / 2, fd.goalWidth / 2, 0.0});

  fieldMarkers.push_back(FieldMarker{'X', 0.0, -fd.circleRadius, 0.0});
  fieldMarkers.push_back(FieldMarker{'X', 0.0, fd.circleRadius, 0.0});

  fieldMarkers.push_back(FieldMarker{'X', fd.length / 2 - fd.penaltyDist, 0.0, 0.0});
  fieldMarkers.push_back(FieldMarker{'X', -fd.length / 2 + fd.penaltyDist, 0.0, 0.0});

  fieldMarkers.push_back(FieldMarker{'T', 0.0, fd.width / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', 0.0, -fd.width / 2, 0.0});

  fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.penaltyAreaLength), fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.penaltyAreaLength), -fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.penaltyAreaLength), fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.penaltyAreaLength), -fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, -fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, fd.penaltyAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, -fd.penaltyAreaWidth / 2, 0.0});

  fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.goalAreaLength), fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.goalAreaLength), -fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.goalAreaLength), fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.goalAreaLength), -fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, -fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, fd.goalAreaWidth / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, -fd.goalAreaWidth / 2, 0.0});

  fieldMarkers.push_back(FieldMarker{'L', fd.length / 2, fd.width / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', fd.length / 2, -fd.width / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', -fd.length / 2, fd.width / 2, 0.0});
  fieldMarkers.push_back(FieldMarker{'L', -fd.length / 2, -fd.width / 2, 0.0});

  // Update mapByType cache
  mapByType.clear();
  for (auto &m : fieldMarkers) {
    mapByType[m.type].push_back(m);
  }
}

void Locator::setPFParams(int numParticles, double initMargin, bool ownHalf, double sensorNoise, std::vector<double> alphas, double alphaSlow, double alphaFast,
                          double injectionRatio, double zeroMotionTransThresh, double zeroMotionRotThresh, bool resampleWhenStopped, double clusterDistThr,
                          double clusterThetaThr, double smoothAlpha, double kldErr, double kldZ, int minParticles, int maxParticles, double resX, double resY,
                          double resTheta, double obsVarX, double obsVarY) {
  this->pfNumParticles = numParticles;
  this->pfInitFieldMargin = initMargin;
  this->pfInitOwnHalfOnly = ownHalf;
  this->pfSensorNoiseR = sensorNoise;
  this->pfAlpha1 = alphas[0];
  this->pfAlpha2 = alphas[1];
  this->pfAlpha3 = alphas[2];
  this->pfAlpha4 = alphas[3];
  this->alpha_slow = alphaSlow;
  this->alpha_fast = alphaFast;
  this->pfInjectionRatio = injectionRatio;
  this->pfZeroMotionTransThresh = zeroMotionTransThresh;
  this->pfZeroMotionRotThresh = zeroMotionRotThresh;
  this->pfResampleWhenStopped = resampleWhenStopped;
  this->pfClusterDistThr = clusterDistThr;
  this->pfClusterThetaThr = clusterThetaThr;
  this->pfSmoothAlpha = smoothAlpha;

  this->kldErr = kldErr;
  this->kldZ = kldZ;
  this->minParticles = minParticles;
  this->maxParticles = maxParticles;
  this->pfResolutionX = resX;
  this->pfResolutionY = resY;
  this->pfResolutionTheta = resTheta;
  this->pfObsVarX = obsVarX;
  this->pfObsVarY = obsVarY;
}

void Locator::init(FieldDimensions fd, int minMarkerCntParam, double residualToleranceParam, double muOffestParam, bool enableLogParam, string logIPParam) {
  fieldDimensions = fd;
  calcFieldMarkers(fd);
  enableLog = enableLogParam;
  logIP = logIPParam;
  // if (enableLog) {
  // logger = &log;
  // auto connectError = log.connect(logIP);
  // if (connectError.is_err()) prtErr(format("Rerun log connect Error: %s", connectError.description.c_str()));
  // auto saveError = log.save("/home/booster/log.rrd");
  // if (saveError.is_err()) prtErr(format("Rerun log save Error: %s", saveError.description.c_str()));
  // }
}

void Locator::globalInitPF(Pose2D currentOdom) {
  double xMin = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
  double xMax = pfInitFieldMargin;
  double yMin = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
  double yMax = fieldDimensions.width / 2.0 + pfInitFieldMargin;
  double thetaMin = -M_PI;
  double thetaMax = M_PI;

  isPFInitialized = true;
  lastPFOdomPose = currentOdom;

  int num = pfNumParticles;
  pfParticles.resize(num);

  // std::srand(std::time(0)); // No longer needed
  for (int i = 0; i < num; i++) {
    pfParticles[i].x = uniformRandom(xMin, xMax);
    pfParticles[i].y = uniformRandom(yMin, yMax);
    double thetaSpread = deg2rad(30.0);
    double thetaCenter;

    if (pfParticles[i].y > 0)
      thetaCenter = -M_PI / 2.0;
    else
      thetaCenter = M_PI / 2.0;

    pfParticles[i].theta = toPInPI(thetaCenter + uniformRandom(-thetaSpread, thetaSpread));
    pfParticles[i].weight = 1.0 / num;
  }

  w_slow = 0.0;
  w_fast = 0.0;

  hasSmoothedPose = false;
}

void Locator::predictPF(Pose2D currentOdomPose) {

  prtWarn(format("[PF][predictPF] enter | initialized=%d | pfN=%zu | odom=(%.2f %.2f %.2f)", isPFInitialized, pfParticles.size(), currentOdomPose.x,
                 currentOdomPose.y, rad2deg(currentOdomPose.theta)));

  if (!isPFInitialized) {
    prtWarn("[PF][predictPF] NOT initialized -> only update lastPFOdomPose");
    lastPFOdomPose = currentOdomPose; // (0,0,0)에서 점프 방지
    globalInitPF(currentOdomPose);
    return;
  }

  double dx = currentOdomPose.x - lastPFOdomPose.x;
  double dy = currentOdomPose.y - lastPFOdomPose.y;
  double dtheta = toPInPI(currentOdomPose.theta - lastPFOdomPose.theta);

  double transDist = sqrt(dx * dx + dy * dy);
  double rotDist = fabs(dtheta);

  double c = cos(lastPFOdomPose.theta);
  double s = sin(lastPFOdomPose.theta);
  double trans_x = c * dx + s * dy; // 로봇좌표계로
  double trans_y = -s * dx + c * dy;
  double rot1 = atan2(trans_y, trans_x);
  double trans = sqrt(trans_x * trans_x + trans_y * trans_y);
  double rot2 = (dtheta - rot1);

  double alpha1 = pfAlpha1;
  double alpha2 = pfAlpha2;
  double alpha3 = pfAlpha3;
  double alpha4 = pfAlpha4;

  for (auto &p : pfParticles) {
    double n_rot1 = rot1 + (gaussianRandom(0, alpha1 * fabs(rot1) + alpha2 * trans));
    double n_trans = trans + (gaussianRandom(0, alpha3 * trans + alpha4 * (fabs(rot1) + fabs(rot2))));
    double n_rot2 = rot2 + (gaussianRandom(0, alpha1 * fabs(rot2) + alpha2 * trans));

    p.x += n_trans * cos(p.theta + n_rot1);
    p.y += n_trans * sin(p.theta + n_rot1);
    p.theta = toPInPI(p.theta + n_rot1 + n_rot2);
  }

  lastPFOdomPose = currentOdomPose;
}

void Locator::correctPF(const vector<FieldMarker> markers) {
  prtWarn(format("[PF][correctPF] enter | initialized=%d | pfN=%zu | markersN=%zu", isPFInitialized, pfParticles.size(), markers.size()));
  if (!isPFInitialized || markers.empty()) {
    prtWarn("[PF][correctPF] not initialized");
    return;
  }

  double sigma = pfSensorNoiseR;
  double totalWeight = 0;
  double avgWeight = 0;

  // Pre-categorize observations by type to avoid repeated looping
  map<char, vector<FieldMarker>> obsByType;
  for (auto &m : markers) {
    obsByType[m.type].push_back(m);
  }

  // Optimization: This could be done once at init if fieldMarkers doesn't change,
  // but it's small enough to do here for safety.

  HungarianAlgorithm hungarian;

  // Weight Update
  for (auto &p : pfParticles) {
    double xMinConstraint = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
    double xMaxConstraint = fieldDimensions.length / 2.0 + pfInitFieldMargin;
    double yMinConstraint = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
    double yMaxConstraint = fieldDimensions.width / 2.0 + pfInitFieldMargin;

    if (p.x < xMinConstraint || p.x > xMaxConstraint || p.y < yMinConstraint || p.y > yMaxConstraint) {
      p.weight = 0.0;
    } else {
      Pose2D pose{p.x, p.y, p.theta};
      double logLikelihood = 0;

      // Process each marker type independently
      for (auto const &[type, obsList] : obsByType) {

        // penalty 타입 걸러주기
        if (mapByType.find(type) == mapByType.end()) continue;

        const auto &mapList = mapByType[type];

        // Construct Cost Matrix (Rows: Obs, Cols: Map)
        int nObs = obsList.size();
        int nMap = mapList.size();

        if (nObs == 0) continue;

        vector<FieldMarker> validObsInField;
        validObsInField.reserve(nObs);

        auto getMahalanobisCost = [&](double dx_f, double dy_f, double theta) {
          double c = cos(theta);
          double s = sin(theta);
          double dx_r = c * dx_f + s * dy_f;
          double dy_r = -s * dx_f + c * dy_f;

          return (dx_r * dx_r) / pfObsVarX + (dy_r * dy_r) / pfObsVarY;
        };
        vector<FieldMarker> obsInField;
        for (auto &m_r : obsList) {
          FieldMarker m_f = markerToFieldFrame(m_r, pose);
          obsInField.push_back(m_f);
        }
        double baseRejectCost = 9.0;
        int nCols = nMap + nObs;

        // Resize reused buffer if needed (or just assign which handles resize)
        flatCostMatrix.assign(nObs * nCols, baseRejectCost);

        for (int i = 0; i < nObs; ++i) {
          for (int j = 0; j < nMap; ++j) {
            double dx = obsInField[i].x - mapList[j].x;
            double dy = obsInField[i].y - mapList[j].y;
            flatCostMatrix[i * nCols + j] = getMahalanobisCost(dx, dy, pose.theta);
          }
        }

        vector<int> assignment;
        double minTotalDistSq = hungarian.Solve(flatCostMatrix, nObs, nCols, assignment);

        double sumCost = 0.0;
        for (int i = 0; i < nObs; ++i) {
          int j = assignment[i];
          if (j < 0) continue;

          if (j < nMap) sumCost += flatCostMatrix[i * nCols + j];
        }
        logLikelihood += -0.5 * sumCost;
      }

      double likelihood = exp(logLikelihood);
      p.weight *= likelihood;
    }
    totalWeight += p.weight;
  }

  if (pfParticles.size() > 0) avgWeight = totalWeight / pfParticles.size();

  // Normalize weights
  if (totalWeight < 1e-10) {
    for (auto &p : pfParticles)
      p.weight = 1.0 / pfParticles.size();
  } else {
    for (auto &p : pfParticles)
      p.weight /= totalWeight;
  }

  double p_inject = this->pfInjectionRatio;

  double sqSum = 0;
  for (auto &p : pfParticles)
    sqSum += p.weight * p.weight;
  double ess = 1.0 / (sqSum + 1e-9);

  if (ess < pfParticles.size() * 0.4) {
    vector<Particle> newParticles;
    newParticles.reserve(maxParticles);

    // Low Variance Sampling
    double r = uniformRandom(0.0, 1.0 / pfParticles.size());
    double c = pfParticles[0].weight;
    int i = 0;

    int targetNum = pfParticles.size();

    for (int m = 0; m < targetNum; m++) {
      double u = r + (double)m / targetNum;
      while (u > c) {
        i = (i + 1) % pfParticles.size();
        c += pfParticles[i].weight;
      }

      Particle newP = pfParticles[i];

      // Injection logic inside loop
      if (uniformRandom(0.0, 1.0) < p_inject) {
        double xMin = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
        double xMax = fieldDimensions.length / 2.0 + pfInitFieldMargin;
        double yMin = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
        double yMax = fieldDimensions.width / 2.0 + pfInitFieldMargin;

        newP.x = uniformRandom(xMin, xMax);
        newP.y = uniformRandom(yMin, yMax);
        newP.theta = toPInPI(uniformRandom(-M_PI, M_PI));
        newP.weight = 1.0;
      }
      newParticles.push_back(newP);
    }
    int M = newParticles.size();
    // Normalize weights for new set
    for (auto &p : newParticles)
      p.weight = 1.0 / M;

    pfParticles = newParticles;
  }
}

Pose2D Locator::getEstimatePF() {
  if (pfParticles.empty()) return {0, 0, 0};

  struct Cluster {
    double totalWeight = 0;
    double xSum = 0;
    double ySum = 0;
    double cosSum = 0;
    double sinSum = 0;
    double leaderX = 0;
    double leaderY = 0;
    double leaderTheta = 0;
  };

  std::vector<Cluster> clusters;

  // Sort
  std::vector<int> sortedIndices(pfParticles.size());
  std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
  std::sort(sortedIndices.begin(), sortedIndices.end(), [&](int a, int b) { return pfParticles[a].weight > pfParticles[b].weight; });

  return {pfParticles[sortedIndices[0]].x, pfParticles[sortedIndices[0]].y, pfParticles[sortedIndices[0]].theta};

  // // Take weighted average of top 10 particles
  // double sumW = 0.0;
  // double x = 0.0, y = 0.0, sumSin = 0.0, sumCos = 0.0;

  // int count = std::min((int)sortedIndices.size(), 10);
  // for (int i = 0; i < count; ++i) {
  //   const auto &p = pfParticles[sortedIndices[i]];
  //   x += p.x * p.weight;
  //   y += p.y * p.weight;
  //   sumSin += sin(p.theta) * p.weight;
  //   sumCos += cos(p.theta) * p.weight;
  //   sumW += p.weight;
  // }

  // if (sumW > 1e-9) {
  //   return {x / sumW, y / sumW, atan2(sumSin, sumCos)};
  // } else {
  //   // Fallback if weights are zero (shouldn't happen with proper normalization)
  //   return {pfParticles[sortedIndices[0]].x, pfParticles[sortedIndices[0]].y, pfParticles[sortedIndices[0]].theta};
  // }

  // // clustering
  // for (int idx : sortedIndices) {
  //   auto &p = pfParticles[idx];
  //   bool added = false;
  //   for (auto &c : clusters) {
  //     // 게이팅
  //     double d = std::hypot(p.x - c.leaderX, p.y - c.leaderY);
  //     double dTheta = std::fabs(toPInPI(p.theta - c.leaderTheta));
  //     // weighted sum 구하기
  //     if (d < pfClusterDistThr && dTheta < pfClusterThetaThr) {
  //       c.totalWeight += p.weight;
  //       c.xSum += p.x * p.weight;
  //       c.ySum += p.y * p.weight;
  //       c.cosSum += cos(p.theta) * p.weight;
  //       c.sinSum += sin(p.theta) * p.weight;
  //       added = true;
  //       break;
  //     }
  //   }
  //   // cluster에 포함되지 않았다면 다른 클러스터의 대장이 됨
  //   if (!added) {
  //     Cluster c;
  //     c.totalWeight = p.weight;
  //     c.xSum = p.x * p.weight;
  //     c.ySum = p.y * p.weight;
  //     c.cosSum = cos(p.theta) * p.weight;
  //     c.sinSum += sin(p.theta) * p.weight;
  //     c.leaderX = p.x;
  //     c.leaderY = p.y;
  //     c.leaderTheta = p.theta;
  //     clusters.push_back(c);
  //   }
  // }

  // // 가장 큰 가중치 합을 가진 클러스터 선택
  // int bestClusterIdx = -1;
  // double maxWeight = -1.0;

  // for (int i = 0; i < clusters.size(); i++) {
  //   if (clusters[i].totalWeight > maxWeight) {
  //     maxWeight = clusters[i].totalWeight;
  //     bestClusterIdx = i;
  //   }
  // }

  // if (bestClusterIdx == -1) return {0, 0, 0};

  // // expected value
  // Pose2D rawEstPose;
  // auto &bestC = clusters[bestClusterIdx];
  // if (bestC.totalWeight > 0) {
  //   rawEstPose = Pose2D{bestC.xSum / bestC.totalWeight, bestC.ySum / bestC.totalWeight, atan2(bestC.sinSum, bestC.cosSum)}; // 기댓값
  // } else {
  //   rawEstPose = Pose2D{bestC.leaderX, bestC.leaderY, bestC.leaderTheta};
  // }

  // // EMA smoothing
  // if (!hasSmoothedPose) {
  //   smoothedPose = rawEstPose;
  //   hasSmoothedPose = true;
  // } else {
  //   smoothedPose.x = pfSmoothAlpha * rawEstPose.x + (1.0 - pfSmoothAlpha) * smoothedPose.x;
  //   smoothedPose.y = pfSmoothAlpha * rawEstPose.y + (1.0 - pfSmoothAlpha) * smoothedPose.y;
  //   double diffTheta = toPInPI(rawEstPose.theta - smoothedPose.theta);
  //   smoothedPose.theta = toPInPI(smoothedPose.theta + pfSmoothAlpha * diffTheta);
  // }

  // return smoothedPose;
}

void Locator::setLog(rerun::RecordingStream *stream) { logger = stream; }

void Locator::logParticles(double time_sec) {
  if (!enableLog || logger == nullptr) return;

  const size_t pfN = pfParticles.size();

  prtWarn(format("[PF][logParticles] pfN=%zu enableLog=%d", pfParticles.size(), enableLog ? 1 : 0));

  std::vector<rerun::Position2D> origins;
  std::vector<rerun::Vector2D> vectors;
  std::vector<rerun::Color> colors;
  std::vector<float> radii;

  origins.reserve(pfN);
  vectors.reserve(pfN);
  colors.reserve(pfN);
  radii.reserve(pfN);

  const float len = 0.1f;

  for (const auto &p : pfParticles) {
    float x0 = static_cast<float>(p.x);
    float y0 = static_cast<float>(p.y);
    float dx = len * std::cos(p.theta);
    float dy = len * std::sin(p.theta);

    origins.push_back({x0, -y0}); // Flip Y for display
    vectors.push_back({dx, -dy}); // Flip Y for display

    // Dynamic Alpha: Base 50, scales up with weight
    // multiplier 1000 ensures that even small weights get some boost, but max out at 200
    uint8_t alpha = static_cast<uint8_t>(std::clamp(50.0 + p.weight * 2000.0, 50.0, 200.0));
    colors.push_back(rerun::Color{0, 255, 255, alpha});

    // Base radius 0.005, max radius 0.05
    float r = 0.005f + 0.045f * (float)(p.weight);
    radii.push_back(r);
  }

  logger->log("field/particles", rerun::Arrows2D::from_vectors(vectors).with_origins(origins).with_colors(colors).with_radii(radii).with_draw_order(19.0));
}

FieldMarker Locator::markerToFieldFrame(FieldMarker marker_r, Pose2D pose_r2f) {
  auto [x, y, theta] = pose_r2f;

  Eigen::Matrix3d transform;
  transform << cos(theta), -sin(theta), x, sin(theta), cos(theta), y, 0, 0, 1;

  Eigen::Vector3d point_r;
  point_r << marker_r.x, marker_r.y, 1.0;

  auto point_f = transform * point_r;

  return FieldMarker{marker_r.type, point_f.x(), point_f.y(), marker_r.confidence};
}

double Locator::minDist(FieldMarker marker) {
  double minDist = std::numeric_limits<double>::infinity();
  double dist;
  for (int i = 0; i < fieldMarkers.size(); i++) {
    auto target = fieldMarkers[i];
    if (target.type != marker.type) { continue; }
    dist = sqrt(pow((target.x - marker.x), 2.0) + pow((target.y - marker.y), 2.0));
    if (dist < minDist) minDist = dist;
  }
  return minDist;
}

// 나중에 모드를 설정해서 initial particle 영역을 달리해야댐
NodeStatus SelfLocateEnterField::tick() {
  if (!brain->locator->getIsPFInitialized()) { brain->locator->globalInitPF(brain->data->robotPoseToOdom); }
  return NodeStatus::SUCCESS;
}

NodeStatus SelfLocate::tick() { return NodeStatus::SUCCESS; }

NodeStatus SelfLocate1M::tick() { return NodeStatus::SUCCESS; }
NodeStatus SelfLocate2X::tick() { return NodeStatus::SUCCESS; }
NodeStatus SelfLocate2T::tick() { return NodeStatus::SUCCESS; }
NodeStatus SelfLocateLT::tick() { return NodeStatus::SUCCESS; }
NodeStatus SelfLocatePT::tick() { return NodeStatus::SUCCESS; }
NodeStatus SelfLocateBorder::tick() { return NodeStatus::SUCCESS; }