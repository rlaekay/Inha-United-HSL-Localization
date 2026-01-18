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

double gaussianRandom(double mean, double stddev) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<double> d(mean, stddev);
  return d(gen);
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
                          double resTheta) {
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

  std::srand(std::time(0));
  for (int i = 0; i < num; i++) {
    pfParticles[i].x = xMin + ((double)rand() / RAND_MAX) * (xMax - xMin);
    pfParticles[i].y = yMin + ((double)rand() / RAND_MAX) * (yMax - yMin);
    double thetaSpread = deg2rad(30.0);
    double thetaCenter;

    if (pfParticles[i].y > 0)
      thetaCenter = -M_PI / 2.0;
    else
      thetaCenter = M_PI / 2.0;

    pfParticles[i].theta = toPInPI(thetaCenter + ((double)rand() / RAND_MAX * 2.0 * thetaSpread) - thetaSpread);
    pfParticles[i].weight = 1.0 / num;
  }

  w_slow = 0.0;
  w_fast = 0.0;

  hasSmoothedPose = false;
}

void Locator::predictPF(Pose2D currentOdomPose) {

  prtWarn(format("[PF][predictPF] enter | initialized=%d | pfN=%zu | "
                 "odom=(%.2f %.2f %.2f)",
                 isPFInitialized, pfParticles.size(), currentOdomPose.x, currentOdomPose.y, rad2deg(currentOdomPose.theta)));

  if (!isPFInitialized) {
    prtWarn("[PF][predictPF] NOT initialized -> only update lastPFOdomPose");
    lastPFOdomPose = currentOdomPose; // (0,0,0)에서 점프 방지
    globalInitPF(currentOdomPose);
    return;
  }

  double dx = currentOdomPose.x - lastPFOdomPose.x;
  double dy = currentOdomPose.y - lastPFOdomPose.y;
  double dtheta = toPInPI(currentOdomPose.theta - lastPFOdomPose.theta);

  // Zero Motion Gate
  double transDist = sqrt(dx * dx + dy * dy);
  double rotDist = fabs(dtheta);

  // if (transDist < pfZeroMotionTransThresh && rotDist < pfZeroMotionRotThresh) {
  //   isRobotMoving = false;
  //   return;
  // }
  isRobotMoving = true;

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

  // Pre-categorize field markers by type
  // Optimization: This could be done once at init if fieldMarkers doesn't change,
  // but it's small enough to do here for safety.

  HungarianAlgorithm hungarian;

  // Weight Update
  for (auto &p : pfParticles) {
    // Check Boundary Constraints
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
        // If map has no markers of this type, these obs are ghosts -> penalty?
        // Current logic: ignore (or implicit penalty by not increasing probability)
        if (mapByType.find(type) == mapByType.end()) continue;

        const auto &mapList = mapByType[type];

        // Construct Cost Matrix (Rows: Obs, Cols: Map)
        int nObs = obsList.size();
        int nMap = mapList.size();

        // Transform all map markers to robot frame? Or obs to field frame?
        // Old logic: markerToFieldFrame (obs -> field). Let's stick to that.
        vector<FieldMarker> obsInField;
        obsInField.reserve(nObs);
        for (auto &m_r : obsList) {
          obsInField.push_back(markerToFieldFrame(m_r, pose));
        }

        vector<vector<double>> costMatrix(nObs, vector<double>(nMap));
        for (int i = 0; i < nObs; ++i) {
          for (int j = 0; j < nMap; ++j) {
            double dx = obsInField[i].x - mapList[j].x;
            double dy = obsInField[i].y - mapList[j].y;
            // Cost = Squared Distance
            costMatrix[i][j] = dx * dx + dy * dy;
          }
        }

        // Solve Assignment
        vector<int> assignment;
        double minTotalDistSq = hungarian.Solve(costMatrix, assignment);

        // Update Likelihood
        // cost is sum of dist^2 for optimal matches
        // For unassigned observations (if nObs > nMap), they are NOT included in cost.
        // We might want to add a penalty for unassigned observations?
        // For now, sticking to "sum of matches" to be safe.
        logLikelihood += -minTotalDistSq / (2 * sigma * sigma);
      }

      double likelihood = exp(logLikelihood);
      p.weight *= likelihood;
    }
    totalWeight += p.weight;
  }

  if (pfParticles.size() > 0) avgWeight = totalWeight / pfParticles.size();

  // Normalize
  if (totalWeight < 1e-10) {
    for (auto &p : pfParticles)
      p.weight = 1.0 / pfParticles.size();
  } else {
    for (auto &p : pfParticles)
      p.weight /= totalWeight;
  }

  double p_inject = 0.1;

  if (isRobotMoving || !pfResampleWhenStopped) {
    double sqSum = 0;
    for (auto &p : pfParticles)
      sqSum += p.weight * p.weight;
    double ess = 1.0 / (sqSum + 1e-9);

    if (ess < pfParticles.size() * 0.5) { // increased threshold slightly
      vector<Particle> newParticles;
      newParticles.reserve(maxParticles); // Reserve max to avoid realloc

      // KLD Sampling Variables
      // We need to track occupied bins
      // using a set for sparsity, key = (x_idx, y_idx, th_idx)
      // std::tuple is slow? let's use a long long key if indices fit
      // x: +/- 10m / 0.2 = +/- 50 -> 100 bins
      // y: +/- 7m / 0.2 = +/- 35 -> 70 bins
      // th: 360 / 10 = 36 bins
      // Key design: ((x + 100) * 1000 + (y + 100)) * 100 + th
      auto getBinKey = [&](const Particle &p) -> long long {
        int xi = (int)floor(p.x / pfResolutionX);
        int yi = (int)floor(p.y / pfResolutionY);
        int thi = (int)floor(p.theta / pfResolutionTheta);
        // Offset indices to be positive for simpler keying (assuming field < 100m)
        return ((long long)(xi + 500) * 1000 + (yi + 500)) * 100 + (thi + 100);
      };

      set<long long> occupiedBins;
      int k = 0;                // num occupied bins
      int M_chi = minParticles; // Initial target count
      int M = 0;                // Current count

      // Low Variance Sampling Setup
      double r = ((double)rand() / RAND_MAX) * (1.0 / pfParticles.size()); // Assuming simplified LV
      // Standard LV requires fixed size. For KLD we often use simple random sampling with probability prop to weight
      // OR we can run LV over the old set and keep adding until we hit M_chi.
      // Let's implement standard "pick proportional to weight" for KLD loop to be strictly correct with dynamic N.
      // But LV is essentially a better version of that.
      // We will perform a simplified LV where we cycle through the cumulative distribution.

      double c = pfParticles[0].weight;
      int i = 0;                              // index of old particle
      double step = 1.0 / pfParticles.size(); // conceptual step size? No, we don't know N yet.

      // KLD Approach: we generate particles one by one until condition met
      // It's often easier to just sample randomly with valid weights.
      // Easiest robust impl: Construct CDF
      vector<double> cdf(pfParticles.size());
      cdf[0] = pfParticles[0].weight;
      for (size_t j = 1; j < pfParticles.size(); ++j)
        cdf[j] = cdf[j - 1] + pfParticles[j].weight;

      do {
        // Select particle
        double u = (double)rand() / RAND_MAX;
        // Binary search for index
        auto it = lower_bound(cdf.begin(), cdf.end(), u);
        int idx = distance(cdf.begin(), it);
        if (idx >= (int)pfParticles.size()) idx = pfParticles.size() - 1;

        Particle newP = pfParticles[idx];

        // Apply motion noise? No, this is resampling step. Motion noise is predict step.
        // Applying minimal jitter? optional. Sticking to copy.

        // Check bin
        long long key = getBinKey(newP);
        if (occupiedBins.find(key) == occupiedBins.end()) {
          occupiedBins.insert(key);
          k++;
          // Recalculate M_chi
          if (k > 1) {
            // Wilson-Hilferty approximation
            double z = kldZ;
            double term = 1.0 - 2.0 / (9.0 * (k - 1)) + sqrt(2.0 / (9.0 * (k - 1))) * z;
            M_chi = (int)ceil((k - 1) / (2.0 * kldErr) * term * term * term);
          }
        }

        newParticles.push_back(newP);
        M++;

      } while ((M < M_chi || M < minParticles) && M < maxParticles);

      // Normalize weights for new set
      for (auto &p : newParticles)
        p.weight = 1.0 / M;

      pfParticles = newParticles;
    }
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

  // clustering
  for (int idx : sortedIndices) {
    auto &p = pfParticles[idx];
    bool added = false;
    for (auto &c : clusters) {
      // 게이팅
      double d = std::hypot(p.x - c.leaderX, p.y - c.leaderY);
      double dTheta = std::fabs(toPInPI(p.theta - c.leaderTheta));
      // weighted sum 구하기
      if (d < pfClusterDistThr && dTheta < pfClusterThetaThr) {
        c.totalWeight += p.weight;
        c.xSum += p.x * p.weight;
        c.ySum += p.y * p.weight;
        c.cosSum += cos(p.theta) * p.weight;
        c.sinSum += sin(p.theta) * p.weight;
        added = true;
        break;
      }
    }
    // cluster에 포함되지 않았다면 다른 클러스터의 대장이 됨
    if (!added) {
      Cluster c;
      c.totalWeight = p.weight;
      c.xSum = p.x * p.weight;
      c.ySum = p.y * p.weight;
      c.cosSum = cos(p.theta) * p.weight;
      c.sinSum += sin(p.theta) * p.weight;
      c.leaderX = p.x;
      c.leaderY = p.y;
      c.leaderTheta = p.theta;
      clusters.push_back(c);
    }
  }

  // 가장 큰 가중치 합을 가진 클러스터 선택
  int bestClusterIdx = -1;
  double maxWeight = -1.0;

  for (int i = 0; i < clusters.size(); i++) {
    if (clusters[i].totalWeight > maxWeight) {
      maxWeight = clusters[i].totalWeight;
      bestClusterIdx = i;
    }
  }

  if (bestClusterIdx == -1) return {0, 0, 0};

  // expected value
  Pose2D rawEstPose;
  auto &bestC = clusters[bestClusterIdx];
  if (bestC.totalWeight > 0) {
    rawEstPose = Pose2D{bestC.xSum / bestC.totalWeight, bestC.ySum / bestC.totalWeight, atan2(bestC.sinSum, bestC.cosSum)}; // 기댓값
  } else {
    rawEstPose = Pose2D{bestC.leaderX, bestC.leaderY, bestC.leaderTheta};
  }

  // EMA smoothing
  if (!hasSmoothedPose) {
    smoothedPose = rawEstPose;
    hasSmoothedPose = true;
  } else {
    smoothedPose.x = pfSmoothAlpha * rawEstPose.x + (1.0 - pfSmoothAlpha) * smoothedPose.x;
    smoothedPose.y = pfSmoothAlpha * rawEstPose.y + (1.0 - pfSmoothAlpha) * smoothedPose.y;
    double diffTheta = toPInPI(rawEstPose.theta - smoothedPose.theta);
    smoothedPose.theta = toPInPI(smoothedPose.theta + pfSmoothAlpha * diffTheta);
  }

  return smoothedPose;
}

void Locator::setLog(rerun::RecordingStream *stream) { logger = stream; }

void Locator::logParticles(double time_sec) {
  if (!enableLog || logger == nullptr) return;

  const size_t pfN = pfParticles.size();

  prtWarn(format("[PF][logParticles] pfN=%zu enableLog=%d", pfParticles.size(), enableLog ? 1 : 0));

  std::vector<std::vector<rerun::Position2D>> lines;
  lines.reserve(pfN);

  const float len = 0.1f;

  for (const auto &p : pfParticles) {
    float x0 = static_cast<float>(p.x);
    float y0 = static_cast<float>(p.y);
    float x1 = x0 + len * std::cos(p.theta);
    float y1 = y0 + len * std::sin(p.theta);

    lines.push_back({{x0, -y0}, {x1, -y1}});
  }

  std::vector<rerun::Color> colors(pfN, rerun::Color{0, 255, 255, 120});

  // 얇은 선
  std::vector<float> radii(pfN, 0.0025f);

  logger->log("field/particles", rerun::LineStrips2D(lines).with_colors(colors).with_radii(radii).with_draw_order(19.0));
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