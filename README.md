# MARK-I

| **Item** | **Description** | **Details** | **Notes** |
|---|---|---|---|
| **Filter** | Adaptive augmented SIR with random injection |  |  |
| **Initialization** | At game start | `selfLocateEnterField()` → `globalInitPF()` |  |
| **Prior** | Odometry-based prior<br>Thrun odometry model | `odometerCallback()` → `predictPF()` |  |
| **Likelihood** | Aggregated marker-wise probabilities | `detectProcessMarkings()` → `correctPF()` |  |
| **Resampling** | ESS-based resampling |  |  |
| **Final pose estimation** | Mean pose of particles |  |  |
| **Number of particles** | 150 |  |  |
| **Current issues** | Likelihood becomes highly unstable when the number of observed markers is small (1–2) |  | Consider enforcing a minimum marker count |
| **Future improvements** | - Measurement smoothing<br>- Enforce a minimum marker count, or<br>- Apply ILM, or<br>- Use Booster’s resampling algorithm<br>- Lower the ESS threshold |  |  |
| **Open questions** | - Real-time fusion vs. fusion after measurement filtering (e.g., ILM)<br>- How to perform objective performance evaluation? |  |  |
| **Parameters** | - Motion model<br>- Measurement model<br>- ESS ratio<br>- Injection rate<br>- `alpha_slow`, `alpha_fast`<br>- Number of particles<br>- Initialization bounds |  |  |

-------------------------------

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
