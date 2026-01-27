# MARK-III
| **항목** | **내용** | **상세** | **비고** |
|---|---|---|---|
| **필터** | adaptive SIR |  |  |
| **init** | game 시작과 동시<br>우리쪽 코트 바깥 & facing inward | `selfLocateEnterField()` → `globalInitPF()` |  |
| **Prior** | **Velocity-based stochastic model** | `odometerCallback()` → `predictPF()` | rot→rot<br>rot→trans<br>trans→trans<br>trans→rot |
| **Likelihood** | **LAP matching (Hungarian algorithm)**<br>**Cost:** suppressed joint likelihood of 2D Gaussian<br>(penalty proportional to distance from robot) | `detectProcessMarkings()` → `correctPF()` | distance sd: 1.0 |
| **Resampling** | ESS resampling with orientation constraint<br>very low threshold due to accurate measurement |  | \< 0.3 N |
| **Pose Extraction** | DBSCAN-like clustering<br>1. weight 정렬 (top 80%)<br>2. DBSCAN으로 클러스터 형성<br>3. weight sum + size 기반 필터링<br>4. previous best vs updated best 비교로 최종 선택 |  |  |
| **파티클 개수** | 150 |  |  |
| **Random injection** | very low (symmetric map) | resampling 단계에서 수행 | 1% |
| **Outlier filtering** | correction step에서 weight = 0 |  |  |
| **파라미터** |  |  |  |
| **당장 문제** |  |  |  |

---------------------------------------------------------------------------------
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
