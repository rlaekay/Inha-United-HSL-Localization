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
                          double clusterThetaThr, double smoothAlpha) {
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
}

void Locator::init(FieldDimensions fd, int minMarkerCntParam, double residualToleranceParam, double muOffestParam, bool enableLogParam, string logIPParam) {
  fieldDimensions = fd;
  calcFieldMarkers(fd);
  enableLog = enableLogParam;
  logIP = logIPParam;
}

void Locator::globalInitPF(Pose2D currentOdom) {
  double xMin = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
  double xMax = pfInitFieldMargin;
  double yMin = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
  double yMax = fieldDimensions.width / 2.0 + pfInitFieldMargin;

  isPFInitialized = true;
  lastPFOdomPose = currentOdom;

  // Initialize Matrix (4 x N)
  pfParticles.resize(4, pfNumParticles);

  // Random Init
  // Row 0: x, Row 1: y, Row 2: theta, Row 3: weight
  std::srand(std::time(0));

  // Use Eigen NullaryExpr or loop. Loop is explicit.
  for (int i = 0; i < pfNumParticles; i++) {
    pfParticles(0, i) = xMin + ((double)rand() / RAND_MAX) * (xMax - xMin);
    pfParticles(1, i) = yMin + ((double)rand() / RAND_MAX) * (yMax - yMin);

    double thetaSpread = deg2rad(30.0);
    double thetaCenter = (pfParticles(1, i) > 0) ? -M_PI / 2.0 : M_PI / 2.0;

    pfParticles(2, i) = toPInPI(thetaCenter + ((double)rand() / RAND_MAX * 2.0 * thetaSpread) - thetaSpread);
    pfParticles(3, i) = 1.0 / pfNumParticles;
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

  // Calculate Standard Deviations (Constant for all particles)
  double std_rot1 = alpha1 * fabs(rot1) + alpha2 * trans;
  double std_trans = alpha3 * trans + alpha4 * (fabs(rot1) + fabs(rot2));
  double std_rot2 = alpha1 * fabs(rot2) + alpha2 * trans;

  // Fully Vectorized Box-Muller Transform
  // We need 3 Gaussian vectors. Box-Muller takes 2 Uniform vectors to make 2 Gaussian vectors.
  // We will generate 4 Uniform vectors to create 4 Gaussian vectors (discard 1, or just use 2 pairs).

  // 1. Generate Uniform Randoms [0, 1]
  // Eigen::Random returns [-1, 1], so we shift and scale.
  Eigen::ArrayXd u1 = (Eigen::ArrayXd::Random(pfNumParticles) + 1.0) * 0.5;
  Eigen::ArrayXd u2 = (Eigen::ArrayXd::Random(pfNumParticles) + 1.0) * 0.5;
  Eigen::ArrayXd u3 = (Eigen::ArrayXd::Random(pfNumParticles) + 1.0) * 0.5;
  Eigen::ArrayXd u4 = (Eigen::ArrayXd::Random(pfNumParticles) + 1.0) * 0.5;

  // Avoid log(0)
  u1 = u1.max(1e-9);
  u3 = u3.max(1e-9);

  // 2. Box-Muller Transform
  // z0 = sqrt(-2 ln u1) * cos(2 pi u2)
  // z1 = sqrt(-2 ln u1) * sin(2 pi u2)
  Eigen::ArrayXd mag1 = (-2.0 * u1.log()).sqrt();
  Eigen::ArrayXd mag2 = (-2.0 * u3.log()).sqrt();

  Eigen::ArrayXd z_rot1 = mag1 * (2.0 * M_PI * u2).cos();
  Eigen::ArrayXd z_trans = mag1 * (2.0 * M_PI * u2).sin();
  Eigen::ArrayXd z_rot2 = mag2 * (2.0 * M_PI * u4).cos();

  // 3. Apply Noise Scale
  Eigen::ArrayXd n_rot1 = rot1 + z_rot1 * std_rot1;
  Eigen::ArrayXd n_trans = trans + z_trans * std_trans;
  Eigen::ArrayXd n_rot2 = rot2 + z_rot2 * std_rot2;

  // Vectorized Update
  // theta' = theta + n_rot1
  // x' = x + n_trans * cos(theta')
  // y' = y + n_trans * sin(theta')
  // theta'' = theta' + n_rot2

  // Access rows as Arrays
  auto x = pfParticles.row(0).array();
  auto y = pfParticles.row(1).array();
  auto theta = pfParticles.row(2).array();

  // Temporary theta_prime
  Eigen::ArrayXd theta_prime = theta + n_rot1; // theta is already Array

  // Update X and Y
  // Note: Eigen arrays support .cos() and .sin() vectorized
  x += n_trans * theta_prime.cos();
  y += n_trans * theta_prime.sin();

  // Update Theta
  theta = theta_prime + n_rot2;

  // Normalize Theta
  // We can try to vectorize this too with simple bounds if close to PI,
  // but toPInPI involves modulo/loops which is hard to vector-op without 'select'.
  // UnaryExpr is the standard way.
  pfParticles.row(2) = pfParticles.row(2).unaryExpr([](double t) { return toPInPI(t); });

  lastPFOdomPose = currentOdomPose;
}

void Locator::correctPF(const vector<FieldMarker> markers) {
  prtWarn(format("[PF][correctPF] enter | initialized=%d | pfN=%d | markersN=%zu", isPFInitialized, pfNumParticles, markers.size()));
  if (!isPFInitialized || markers.empty()) {
    prtWarn("[PF][correctPF] not initialized");
    return;
  }

  double sigma = pfSensorNoiseR;
  double newTotalWeight = 0;

  // Pre-categorize observations by type
  map<char, vector<FieldMarker>> obsByType;
  for (auto &m : markers) {
    obsByType[m.type].push_back(m);
  }

  // Indices Sort for Heuristic Optimization
  // We want to process high-weight particles first for the heuristic
  std::vector<int> pIndices(pfNumParticles);
  std::iota(pIndices.begin(), pIndices.end(), 0);
  std::sort(pIndices.begin(), pIndices.end(), [&](int a, int b) { return pfParticles(3, a) > pfParticles(3, b); });

  // Optimization Parameters
  const int TOP_N = 10;
  const int RANDOM_M = 10;
  std::vector<bool> doFullCalc(pfNumParticles, false);

  // Select Top N
  for (int i = 0; i < std::min(pfNumParticles, TOP_N); ++i) {
    doFullCalc[pIndices[i]] = true;
  }
  // Select Random M
  for (int i = 0; i < RANDOM_M; ++i) {
    int idx = rand() % pfNumParticles;
    doFullCalc[idx] = true;
  }

  std::map<char, std::vector<int>> consensusAssignments;
  HungarianAlgorithm hungarian;

  // 1. Establish Consensus (First Pass on Top Particle)
  for (auto const &[type, obsList] : obsByType) {
    if (mapByType.find(type) == mapByType.end()) continue;
    const auto &mapList = mapByType[type];

    if (pfNumParticles > 0) {
      int bestIdx = pIndices[0];
      Pose2D pose{pfParticles(0, bestIdx), pfParticles(1, bestIdx), pfParticles(2, bestIdx)};
      int nObs = obsList.size();
      int nMap = mapList.size();

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
          costMatrix[i][j] = dx * dx + dy * dy;
        }
      }

      vector<int> assignment;
      hungarian.Solve(costMatrix, assignment);
      consensusAssignments[type] = assignment;
    }
  }

  // 2. Weight Update Loop
  // We iterate using indices to check boundary conditions
  // Optimization: Pre-calculate Likelihood vector?
  // Since Hungarian logic is complex inside loop, we stick to loop.

  Eigen::VectorXd likelihoods(pfNumParticles);

  for (int i = 0; i < pfNumParticles; ++i) {
    double px = pfParticles(0, i);
    double py = pfParticles(1, i);
    double ptheta = pfParticles(2, i);

    // Boundary Check
    double xMinConstraint = -fieldDimensions.length / 2.0 - pfInitFieldMargin;
    double xMaxConstraint = fieldDimensions.length / 2.0 + pfInitFieldMargin;
    double yMinConstraint = -fieldDimensions.width / 2.0 - pfInitFieldMargin;
    double yMaxConstraint = fieldDimensions.width / 2.0 + pfInitFieldMargin;

    if (px < xMinConstraint || px > xMaxConstraint || py < yMinConstraint || py > yMaxConstraint) {
      pfParticles(3, i) = 0.0;
      likelihoods(i) = 0.0; // effectively kills it
    } else {
      Pose2D pose{px, py, ptheta};
      double logLikelihood = 0;

      for (auto const &[type, obsList] : obsByType) {
        if (mapByType.find(type) == mapByType.end()) continue;
        const auto &mapList = mapByType[type];
        int nObs = obsList.size();
        int nMap = mapList.size();

        vector<FieldMarker> obsInField;
        obsInField.reserve(nObs);
        for (auto &m_r : obsList) {
          obsInField.push_back(markerToFieldFrame(m_r, pose));
        }

        double minTotalDistSq = 0;

        if (doFullCalc[i]) {
          vector<vector<double>> costMatrix(nObs, vector<double>(nMap));
          for (int r = 0; r < nObs; ++r) {
            for (int c = 0; c < nMap; ++c) {
              double dx = obsInField[r].x - mapList[c].x;
              double dy = obsInField[r].y - mapList[c].y;
              costMatrix[r][c] = dx * dx + dy * dy;
            }
          }
          vector<int> assignment;
          minTotalDistSq = hungarian.Solve(costMatrix, assignment);
        } else {
          const vector<int> &assignment = consensusAssignments[type];
          for (int r = 0; r < nObs; ++r) {
            int c = assignment[r];
            if (c >= 0 && c < nMap) {
              double dx = obsInField[r].x - mapList[c].x;
              double dy = obsInField[r].y - mapList[c].y;
              minTotalDistSq += dx * dx + dy * dy;
            }
          }
        }
        logLikelihood += -minTotalDistSq / (2 * sigma * sigma);
      }
      likelihoods(i) = exp(logLikelihood);
    }
  }

  // Batch Update Weights
  pfParticles.row(3).array() *= likelihoods.array();

  // Calculate Total Weight
  double totalWeight = pfParticles.row(3).sum();
  double avgWeight = 0;
  if (pfNumParticles > 0) avgWeight = totalWeight / pfNumParticles;

  // Normalize
  if (totalWeight < 1e-10) {
    pfParticles.row(3).fill(1.0 / pfNumParticles);
    totalWeight = 1.0;
  } else {
    pfParticles.row(3) /= totalWeight;
  }

  // Resampling Logic (Low Variance Resampling)
  // Only resample if ESS is low or moving
  double sqSum = pfParticles.row(3).array().square().sum();
  double ess = 1.0 / (sqSum + 1e-9);

  if ((isRobotMoving || !pfResampleWhenStopped) && ess < pfNumParticles * 0.5) {
    Eigen::MatrixXd newParticles(4, pfNumParticles);
    double r = ((double)rand() / RAND_MAX) / pfNumParticles;
    double c = pfParticles(3, 0);
    int i = 0;
    for (int m = 0; m < pfNumParticles; ++m) {
      double U = r + (double)m * (1.0 / pfNumParticles);
      while (U > c && i < pfNumParticles - 1) {
        i++;
        c += pfParticles(3, i);
      }
      newParticles.col(m) = pfParticles.col(i);
      newParticles(3, m) = 1.0 / pfNumParticles; // Reset weight
    }
    pfParticles = newParticles;
    totalWeight = 1.0; // Reset total weight after normalization
  }
}

Pose2D Locator::getEstimatePF() {
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

  // return {pfParticles[sortedIndices[0]].x, pfParticles[sortedIndices[0]].y, pfParticles[sortedIndices[0]].theta};

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

  Pose2D Locator::getEstimatePF() {
    if (pfNumParticles == 0) return {0, 0, 0};

    double meanX = (pfParticles.row(0).array() * pfParticles.row(3).array()).sum();
    double meanY = (pfParticles.row(1).array() * pfParticles.row(3).array()).sum();

    double meanSin = (pfParticles.row(2).array().sin() * pfParticles.row(3).array()).sum();
    double meanCos = (pfParticles.row(2).array().cos() * pfParticles.row(3).array()).sum();
    double meanTheta = atan2(meanSin, meanCos);

    Pose2D result = {meanX, meanY, meanTheta};

    // Smoothing (EMA)
    if (!hasSmoothedPose) {
      smoothedPose = result;
      hasSmoothedPose = true;
    } else {
      // Current Estimate
      double diffX = result.x - smoothedPose.x;
      double diffY = result.y - smoothedPose.y;
      double diffTheta = toPInPI(result.theta - smoothedPose.theta);

      smoothedPose.x += pfSmoothAlpha * diffX;
      smoothedPose.y += pfSmoothAlpha * diffY;
      smoothedPose.theta = toPInPI(smoothedPose.theta + pfSmoothAlpha * diffTheta);
    }

    return smoothedPose;
  }

  void Locator::setLog(rerun::RecordingStream * stream) { logger = stream; }

  void Locator::logParticles(double time_sec) {
    if (!enableLog || logger == nullptr) return;

    prtWarn(format("[PF][logParticles] pfN=%d enableLog=%d", pfNumParticles, enableLog ? 1 : 0));

    std::vector<rerun::Position2D> origins;
    std::vector<rerun::Vector2D> vectors;
    std::vector<rerun::Color> colors;
    std::vector<float> radii;

    origins.reserve(pfNumParticles);
    vectors.reserve(pfNumParticles);
    colors.reserve(pfNumParticles);
    radii.reserve(pfNumParticles);

    const float len = 0.1f;

    for (int i = 0; i < pfNumParticles; ++i) {
      float x0 = static_cast<float>(pfParticles(0, i));
      float y0 = static_cast<float>(pfParticles(1, i));
      float th = static_cast<float>(pfParticles(2, i));
      float w = static_cast<float>(pfParticles(3, i));

      float dx = len * std::cos(th);
      float dy = len * std::sin(th);

      origins.push_back({x0, -y0}); // Flip Y for display
      vectors.push_back({dx, -dy}); // Flip Y for display

      // Dynamic Alpha: Base 50, scales up with weight
      // multiplier 1000 ensures that even small weights get some boost, but max out at 200
      uint8_t alpha = static_cast<uint8_t>(std::clamp(50.0 + w * 2000.0, 50.0, 200.0));
      colors.push_back(rerun::Color{0, 255, 255, alpha});

      // Base radius 0.005, max radius 0.05
      float r = 0.005f + 0.08f * w;
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