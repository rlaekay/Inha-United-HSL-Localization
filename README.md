# MARK-III
| **항목** | **내용** | **상세** | 비고 |
| --- | --- | --- | --- |
| **필터** | adaptive SIR |  |  |
| **init** | game 시작과 동시
+ 우리쪽 코트 바깥 & facing inward | selfLocateEnterField() → globalInitPF() |  |
| **Prior** | **Velocity based stochastic Model** | odometerCallback() → predictPF() | rot→rot
rot→trans
trans→trans
trans→rot |
| **Likelihood** | **-LAP matching using hungarian algorithm
-Cost calculation: suppressed joint likelihood of 2D gaussian (disadvantage propotional to the distance from the robot)** | detectrocessMarkings() → correctPF() | distance sd: 1.0 |
| **Resampling** | ESS resampling with orientation constraint
extremely low threshold due to very accurate measurement |  | < 0.3 N |
| **Pose Extraction** | dbscan like clustering
1. weight 정렬. top 80%만 사용
2. sorted particle을 DB scan으로 클러스터 형성
3. weight sum과 size로 cluster filter
4. [지금까지 best vs updated best] cluster로 최종 클러스터 결정 |  |  |
| **파티클 개수** | 150 |  |  |
|  |  |  |  |
| **Random injection** | very low due to symmetric map | done with resampling | 1% |
| **outlier filtering** | 0 weight in correction step |  |  |
| **파라미터** |  |  |  |
| **당장 문제**  |  |  |  |
|  |  |  |  |

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
