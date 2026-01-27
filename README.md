# MARK-IV
| **Item** | **Description** | **Details (Code-level)** | **Notes** |
|---|---|---|---|
| **Filter** | Adaptive SIR particle filter | Explicit ESS-based resampling + clustering-based pose extraction | No explicit augmented PF state |
| **Initialization** | Global initialization at game start | `SelfLocateEnterField::tick()` → `globalInit()` | BT entry-triggered |
| **Initial particle distribution** | Half-field biased initialization | Particles uniformly sampled in own half<br>Two symmetric orientation modes (±90° ± 30°) | Handles initial heading ambiguity |
| **Number of particles** | 150 | `numParticles` | Configurable |
| **Prior (Motion model)** | Thrun odometry model | `predict()`<br>Velocity-based decomposition: rot1–trans–rot2 | State-dependent noise |
| **Zero-motion gate** | Skip prediction if motion is negligible | Early return if<br>`dx, dy < 1e-4` and `dθ < 0.5°` | Prevents particle diffusion |
| **Measurement model** | Marker-based observation model | Observed markers transformed to field frame per particle | SE(2) transform |
| **Data association** | LAP matching (Hungarian algorithm) | Cost matrix = squared Euclidean distance (type-gated) | One-to-one assignment |
| **Likelihood** | Suppressed joint Gaussian likelihood | Mahalanobis-like distance in (normal, perpendicular) frame | Distance-dependent variance |
|  |  | `σ_n = invNormVar · dist + 0.05`<br>`σ_p = invPerpVar · dist + 0.05` | Range-adaptive uncertainty |
| **Weight update** | Exponential likelihood weighting | `w ← w · exp(-0.5 · ΣD² · likelihoodWeight)` | Log-likelihood scaled |
| **Outlier handling** | Hard rejection | Particles outside field bounds → weight = 0 | Geometric constraint |
| **Orientation gating** | Pose-consistency gating | If `|θ_p − θ_est| > orientationGatingThr` → skip update | Only after pose is established |
| **Normalization** | Weight normalization with fallback | Uniform reset if total weight ≈ 0 | Degeneracy protection |
| **Resampling** | ESS-based multinomial resampling | Triggered if `ESS < N · essThreshold` | CDF sampling |
| **Random injection** | ❌ Not explicitly used | No explicit random particle injection | Stability-oriented |
| **Pose extraction (initial)** | Maximum-weight particle | `findBestWeight()` | Used before pose is available |
| **Pose extraction (steady)** | Cluster-based weighted mean | `clusterParticles()` around previous pose | Temporal consistency |
| **Clustering criteria** | Distance + orientation gating | `dist² < clusterDistThr`<br>`|Δθ| < clusterThetaThr` | Local hypothesis validation |
| **Cluster validation** | Minimum total weight | If `Σw < clusterMinWeight` → fallback to max-weight particle | Avoids weak clusters |
| **Final pose estimate** | Cluster-weighted mean + EMA | Position + circular mean of orientation | Smooth output |
| **Smoothing** | EMA (Exponential Moving Average) | `smoothAlpha` applied to pose update | Temporal filtering |
| **Current strengths** | Strong robustness to symmetry & noise | Distance-aware likelihood + pose-referenced clustering | Designed for RoboCup field |
| **Known limitations** | No explicit random recovery | Global recovery relies on re-init only | Could stall in rare failures |
| **Key parameters** | Motion, measurement, clustering, ESS | `α1–α4`, `invNormVar`, `invPerpVar`, `essThreshold`, cluster gates | YAML-configurable |
## Updates
- no random injections
- angular gates enhanced

## In Future
- Initial pose estimation needs to be improved
- Parallel computing to efficiently compute within scarce resources

-----------------------------------------------------------------------------

# K1_Robocup Demo
## introduction
The Booster K1 Robocup official demo allows the robot to make autonomous decisions to kick the ball and complete the full Robocup match. It includes three programs: vision, brain, and game_controller.

-   vision
    -   The Robocup vision recognition program, based on Yolo-v8, detects objects such as robots, soccer balls, and the field, and calculates their positions in the robot's coordinate system using geometric relationships.
-   brain
    -   The Robocup decision-making program reads visual data and GameController game control data, integrates all available information, makes judgments, and controls the robot to perform corresponding actions, completing the match process.
-   game_controller
    -   Reads the game control data packets broadcast by the referee machine on the local area network, converts them into ROS2 topic messages, and makes them available for the brain to use.

##  Install extra dependency
sudo apt-get install ros-humble-backward-ros

## Note
This repo support jetpack 6.2. Adapted to the default TRT model in src/vision/config/vision.yaml.

vision.yaml for jetpack 6.2 machine

    detection_model:
	    model_path: ./src/vision/model/best_digua_second_10.3.engine
	    confidence_threshold: 0.2

## Build and Run

    #Build the programs
    ./scripts/build.sh
    
    #Run on the actual robot
    ./scripts/start.sh

## Documents

[Chinese Version](https://booster.feishu.cn/wiki/SoJCwyIpiiXrp0kgVnKc5rIrn3f)
[English version](https://booster.feishu.cn/wiki/CQXowElA0iy2hhkmPJmcY0wwnHf?renamingWikiNode=false)
