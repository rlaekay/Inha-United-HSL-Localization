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
  static std::normal_distribution<double> N01(0.0, 1.0);
  return mean + stddev * N01(getRandomEngine());
}

double uniformRandom(double min, double max) {
  static std::uniform_real_distribution<double> U01(0.0, 1.0);
  return min + (max - min) * U01(getRandomEngine());
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
                          double resTheta, double invObsVarX, double invObsVarY, double unmatchedPenaltyConfThr, double pfEssThreshold) {
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
  this->invPfObsVarX = invObsVarX;
  this->invPfObsVarY = invObsVarY;
  this->pfUnmatchedPenaltyConfThr = unmatchedPenaltyConfThr;
  this->pfEssThreshold = pfEssThreshold;
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

uint64_t Locator::getBinKey(const Particle &p) {
  // Spatial Hashing
  // x, y mapped to indices based on resolution
  // theta mapped to index
  // key = (xIdx << 42) | (yIdx << 21) | thIdx
  // Assuming 21 bits per axis is enough (2M units)

  // Offset to ensure positive indices
  double offX = fieldDimensions.length / 2.0 + pfInitFieldMargin + 5.0; // Margin
  double offY = fieldDimensions.width / 2.0 + pfInitFieldMargin + 5.0;

  uint64_t xIdx = (uint64_t)((p.x + offX) / pfResolutionX);
  uint64_t yIdx = (uint64_t)((p.y + offY) / pfResolutionY);
  uint64_t thIdx = (uint64_t)(toPInPI(p.theta) / pfResolutionTheta); // 0~2PI

  return (xIdx << 42) | (yIdx << 21) | thIdx;
}

int Locator::calcKLDTarget(int k) {
  if (k <= 1) return minParticles;

  // Wilson-Hilferty approximation
  // n = (k-1) / (2 * epsilon) * { 1 - 2/(9(k-1)) + z_alpha * sqrt(2/(9(k-1))) }^3

  double k_1 = (double)(k - 1);
  double term1 = 1.0 - 2.0 / (9.0 * k_1);
  double term2 = kldZ * sqrt(2.0 / (9.0 * k_1));
  double term = term1 + term2;

  double n = (k_1 / (2.0 * kldErr)) * (term * term * term);

  return (int)ceil(n);
}

void Locator::globalInitPF(Pose2D currentOdom) {
  double xMin = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
  double xMax = pfInitFieldMargin;
  double yMin = fieldDimensions.width / 2.0 - pfInitFieldMargin;
  double yMax = fieldDimensions.width / 2.0 + pfInitFieldMargin;
  double thetaMin = -M_PI;
  double thetaMax = M_PI;

  isPFInitialized = true;
  lastPFOdomPose = currentOdom;

  int num = pfNumParticles;
  pfParticles.resize(num);
  double thetaSpread = deg2rad(30.0);

  for (int i = 0; i < num / 2; i++) {
    pfParticles[i].x = uniformRandom(xMin, xMax);
    pfParticles[i].y = uniformRandom(yMin, yMax);
    double thetaCenter = -M_PI / 2.0;

    pfParticles[i].theta = toPInPI(thetaCenter + uniformRandom(-thetaSpread, thetaSpread));
    pfParticles[i].weight = 1.0 / num;
  }
  for (int i = num / 2; i < num; i++) {
    pfParticles[i].x = uniformRandom(xMin, xMax);
    pfParticles[i].y = -uniformRandom(yMin, yMax);
    double thetaCenter = M_PI / 2.0;
    pfParticles[i].theta = toPInPI(thetaCenter + uniformRandom(-thetaSpread, thetaSpread));
    pfParticles[i].weight = 1.0 / num;
  }
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

  double rot1_dev = alpha1 * fabs(rot1) + alpha2 * trans + 0.0001;
  double trans_dev = alpha3 * trans + alpha4 * (fabs(rot1) + fabs(rot2)) + 0.0001;
  double rot2_dev = alpha1 * fabs(rot2) + alpha2 * trans + 0.0001;

  for (auto &p : pfParticles) {
    double n_rot1 = rot1 + (gaussianRandom(0, rot1_dev));
    double n_trans = trans + (gaussianRandom(0, trans_dev));
    double n_rot2 = rot2 + (gaussianRandom(0, rot2_dev));

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

  double totalWeight = 0;
  double maxWeight = -1.0;

  double invVarX = this->invPfObsVarX;
  double invVarY = this->invPfObsVarY;

  // 1. Weight Update
  for (auto &p : pfParticles) {
    double xMinConstraint = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
    double xMaxConstraint = fieldDimensions.length / 2.0 + pfInitFieldMargin;
    double yMinConstraint = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
    double yMaxConstraint = fieldDimensions.width / 2.0 + pfInitFieldMargin;

    if (p.x < xMinConstraint || p.x > xMaxConstraint || p.y < yMinConstraint || p.y > yMaxConstraint) {
      p.weight = 0.0;
    } else {
      Pose2D pose{p.x, p.y, p.theta};

      double c = cos(pose.theta);
      double s = sin(pose.theta);

      int nObs = markers.size();
      int nMap = fieldMarkers.size();
      int nCols = nMap + nObs;

      obsInFieldBuf.clear();
      obsInFieldBuf.reserve(nObs);

      for (const auto &m_r : markers) {
        double ox = c * m_r.x - s * m_r.y + pose.x;
        double oy = s * m_r.x + c * m_r.y + pose.y;
        obsInFieldBuf.push_back(FieldMarker{m_r.type, ox, oy, m_r.confidence});
      }

      flatCostMatrix.assign(nObs * nCols, baseRejectCost);

      auto getMahalanobisCost = [&](double dx_f, double dy_f) {
        double dx_r = c * dx_f + s * dy_f;
        double dy_r = -s * dx_f + c * dy_f;
        return (dx_r * dx_r) * invVarX + (dy_r * dy_r) * invVarY;
      };

      for (int i = 0; i < nObs; ++i) {
        char obsType = obsInFieldBuf[i].type;
        for (int j = 0; j < nMap; ++j) {
          // Type Mismatch Check
          if (fieldMarkers[j].type != obsType) {
            flatCostMatrix[i * nCols + j] = std::numeric_limits<double>::max();
            continue;
          }

          double dx = obsInFieldBuf[i].x - fieldMarkers[j].x;
          double dy = obsInFieldBuf[i].y - fieldMarkers[j].y;
          flatCostMatrix[i * nCols + j] = getMahalanobisCost(dx, dy);
        }
      }

      vector<int> assignment;
      hungarian.Solve(flatCostMatrix, nObs, nCols, assignment);

      double sumCost = 0.0;
      for (int i = 0; i < nObs; ++i) {
        int j = assignment[i];
        if (j < 0) continue; // Should not happen

        if (j < nMap) {
          sumCost += flatCostMatrix[i * nCols + j];
        } else {
          if (obsInFieldBuf[i].confidence > this->pfUnmatchedPenaltyConfThr) { sumCost += flatCostMatrix[i * nCols + j]; }
        }
      }
      double logLikelihood = -0.5 * sumCost;

      double likelihood = exp(logLikelihood);
      p.weight *= likelihood;
    }
    totalWeight += p.weight;
    if (p.weight > maxWeight) maxWeight = p.weight;
  }

  // Normalization
  if (totalWeight < 1e-10) {
    for (auto &p : pfParticles)
      p.weight = 1.0 / pfParticles.size();
    totalWeight = 1.0;
  } else {
    for (auto &p : pfParticles)
      p.weight /= totalWeight;
  }

  // Updated ESS calculation for Dynamic K logic in getEstimatePF (stored implicitly in weights)
  // For KLD step, we use resample threshold
  double sqSum = 0;
  for (auto &p : pfParticles)
    sqSum += p.weight * p.weight;
  double ess = 1.0 / (sqSum + 1e-9);

  // KLD Resampling
  if (ess < pfParticles.size() * pfEssThreshold || pfParticles.size() < minParticles) {
    vector<Particle> newParticles;
    newParticles.reserve(maxParticles);

    // 2. CDF Construction for Multinomial Sampling
    // Since M (target size) varies, we can't use systematic resampling easily (step size depends on M)
    // We use CDF binary search which is O(log N) per sample.
    std::vector<double> cdf(pfParticles.size());
    cdf[0] = pfParticles[0].weight;
    for (size_t i = 1; i < pfParticles.size(); ++i) {
      cdf[i] = cdf[i - 1] + pfParticles[i].weight;
    }
    // Ensure last is 1.0 (handle float errors)
    cdf.back() = 1.0;

    // KLD State
    std::unordered_set<uint64_t> bins;
    int k_bins = 0;           // Number of non-empty bins
    int M_chi = minParticles; // Target number of particles (adaptive)
    int M_generated = 0;

    // Helper to pick from CDF
    auto sampleCDF = [&]() -> const Particle & {
      double r = uniformRandom(0.0, 1.0);
      auto it = std::upper_bound(cdf.begin(), cdf.end(), r);
      int idx = std::distance(cdf.begin(), it);
      if (idx >= pfParticles.size()) idx = pfParticles.size() - 1;
      return pfParticles[idx];
    };

    // Adaptive Loop
    // Continue while we have fewer particles than target M_chi
    // OR fewer than minParticles
    // AND haven't exceeded maxParticles

    while ((M_generated < M_chi || M_generated < minParticles) && M_generated < maxParticles) {
      Particle pNew;

      // Random Injection (Augmented MCL logic can be integrated here)
      // Check injection ratio
      if (uniformRandom(0.0, 1.0) < pfInjectionRatio) {
        double xMin = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
        double xMax = fieldDimensions.length / 2.0 + pfInitFieldMargin;
        double yMin = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
        double yMax = fieldDimensions.width / 2.0 + pfInitFieldMargin;

        pNew.x = uniformRandom(xMin, xMax);
        pNew.y = uniformRandom(yMin, yMax);
        pNew.theta = toPInPI(uniformRandom(-M_PI, M_PI));
        pNew.weight = 1.0; // Reset weight
      } else {
        // Resample from CDF
        pNew = sampleCDF();
        pNew.weight = 1.0; // Reset weight
      }

      // Add noise (optional, but good for KLD exploration)
      // If we don't add noise here, KLD might under-estimate if particles stack perfectly
      // KLD assumes samples are drawn from the proposal, which usually includes motion noise.
      // Since we are resampling *after* correction, we are essentially building the belief for the *next* step.
      // Usually KLD is done *after* prediction+correction? No, KLD determines how many samples to keep for the *next* prior.

      newParticles.push_back(pNew);
      M_generated++;

      // Update KLD bins
      uint64_t key = getBinKey(pNew);
      if (bins.find(key) == bins.end()) {
        bins.insert(key);
        k_bins++;
        if (k_bins > 1) { M_chi = calcKLDTarget(k_bins); }
      }
    }

    // Normalize weights for new set
    double w_uni = 1.0 / newParticles.size();
    for (auto &p : newParticles)
      p.weight = w_uni;

    pfParticles = newParticles;
    prtWarn(format("[PF] KLD Resample: k_bins=%d -> M_chi=%d | New Size=%zu", k_bins, M_chi, pfParticles.size()));
  }
}

Pose2D Locator::getEstimatePF() {
  if (pfParticles.empty()) return {0, 0, 0};

  // 1. Filter Outliers: Top-K% Selection
  // We only consider the top 10% of particles by weight to ignore the long tail of noise.
  double keep_ratio = 0.07;
  size_t N = pfParticles.size();
  size_t K = std::max<size_t>(1, N * keep_ratio);

  // nth_element puts the top K elements in [0, K)
  // Note: We want LARGER weights, so we use a.weight > b.weight as the comparator.
  std::nth_element(pfParticles.begin(), pfParticles.begin() + K, pfParticles.end(), [](const Particle &a, const Particle &b) { return a.weight > b.weight; });

  // 2. Select Best Particle
  // Find the particle with the maximum weight among the top K.
  auto bestIt = std::max_element(pfParticles.begin(), pfParticles.begin() + K, [](const Particle &a, const Particle &b) { return a.weight < b.weight; });

  Pose2D raw = {bestIt->x, bestIt->y, bestIt->theta};

  // 3. EMA Smoothing
  if (!hasSmoothedPose) {
    smoothedPose = raw;
    hasSmoothedPose = true;
  } else {
    // fast alpha for responsiveness (configured in yaml)
    double alpha = pfSmoothAlpha;

    smoothedPose.x = alpha * raw.x + (1.0 - alpha) * smoothedPose.x;
    smoothedPose.y = alpha * raw.y + (1.0 - alpha) * smoothedPose.y;

    // Angular EMA with wrap-around safety
    double dTheta = toPInPI(raw.theta - smoothedPose.theta);
    smoothedPose.theta = toPInPI(smoothedPose.theta + alpha * dTheta);
  }

  return smoothedPose;
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