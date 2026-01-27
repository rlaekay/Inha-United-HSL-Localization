# MARK-II
| **Item** | **Description** | **Details** | **Notes** |
|---|---|---|---|
| **Filter** | Adaptive SIR |  |  |
| **Initialization** | At game start<br>Own half, facing inward | `selfLocateEnterField()` → `globalInitPF()` |  |
| **Prior** | Thrun odometry model<br>(state-dependent odometry noise model) | `odometerCallback()` → `predictPF()` | rot→rot<br>rot→trans<br>trans→trans<br>trans→rot |
| **Likelihood** | Joint likelihood of markers<br>1D Gaussian | `detectProcessMarkings()` → `correctPF()` | distance sd: 1.0 |
| **Resampling** | ESS-based resampling |  | \< 0.3 N |
| **Estimated pose** | Highest-weight cluster mean + EMA | Weighted mean pose of the cluster around the particle with the highest weight | cluster gate<br>distance: 0.5 m<br>angle: 20 deg |
| **Number of particles** | 150 |  |  |
| **Zero-motion gate** | If the robot is stationary, skip prediction and resampling |  | translation ≤ 5 cm<br>rotation ≤ 3 deg |
| **Random injection** | Very low due to symmetric field | Applied during resampling | 1% |
| **Outlier filtering** | Assign zero weight during correction step |  |  |
| **Parameters** | - Motion model<br>- Measurement model<br>- ESS ratio<br>- Injection rate<br>- Number of particles<br>- Initialization bounds<br>- Cluster distance gate<br>- Cluster angle gate<br>- EMA alpha<br>- Zero-motion translation threshold<br>- Zero-motion rotation threshold |  |  |
| **Current issues** | Estimated pose is unstable |  |  |
| **Future improvements** | Use directional markers (e.g., arrows) for better heading observability<br>Add weight–cluster size coupling<br><br>Include weight as a clustering criterion<br>Consider averaging the top 10 highest-weight particles<br><br>Add input-correlated noise via the Q matrix |  |  |


----------------------------------
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
