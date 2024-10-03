# Robustifying Human-Robot Collaboration through a Hierarchical and Multimodal Framework
![Architecture](/assets/images/architecture.png)

## Abstract
Human-robot collaboration (HRC) is essential for creating flexible manufacturing systems in industries and developing intelligent service robots for everyday applications. Despite its potential, making HRC systems more robust remains a significant challenge. 
In this paper, we introduce a novel **multimodal and hierarchical** framework designed to make human-robot collaboration **more efficient and robust**. By integrating visual observations with speech commands, our system enables more natural and flexible interactions between humans and robots. Additionally, our hierarchical approach for human detection and intention prediction enhances the system's robustness, allowing robots to better understand human behaviors. This proactive understanding enables robots to take timely and appropriate actions based on predicted human intentions. We tested our framework using the KINOVA GEN3 robot in real-world experiments and did extensive user studies. The results demonstrate that our approach effectively improves the efficiency, flexibility, and adaptation ability of HRC, showcasing the framework's potential to significantly improve the way humans and robots work together.

***


This is the official code repo for the paper [Robustifying Human-Robot Collaboration through a Hierarchical and Multimodal Framework](). The video demo can be found at [https://www.youtube.com/watch?v=2kkANN9ueVY](https://www.youtube.com/watch?v=2kkANN9ueVY).

Youtube Video:
[![](https://i.ytimg.com/vi/2kkANN9ueVY/maxresdefault.jpg)](https://www.youtube.com/watch?v=2kkANN9ueVY "")


## Environment
### Hardware
- KINOVA GEN3 robot arm
- OAK-D Lite RGB-D camera
- Microphone

### Software
- ubuntu == 20.04
- python == 3.9
- ros

To set up the environment, run:
```
conda create -n hrc -f env_ubuntu2004.yml

conda activate hrc
```

## Prerequisite
For speech model and scorer, download [deepspeech-0.9.3-models.pbmm] and [deepspeech-0.9.3-models.scorer] from [https://github.com/mozilla/DeepSpeech/releases](https://github.com/mozilla/DeepSpeech/releases), and put these two models into the ./speech.

For robot controller, please set up as [https://github.com/intelligent-control-lab/robot_controller_ros](https://github.com/intelligent-control-lab/robot_controller_ros).

To activate the robot controller, run:
```
source devel/setup.bash

roslaunch kinova kinova_bringup.launch

rosrun rqt_gui rqt_gui

roscore
```
Once the robot is activated, you can see the robot arm moving.

## Usage
```
python controller/receiver.py # start the robot receiving process, waiting for receiving visual and audio signal

# Attention: the three digital numbers succeding [task_id] is necessary!
python run.py --show --task [task_id001] # open up the camera and start the HRC pipeline

python speech/speech_recognize.py # open the speech recognition program and communicate signals to the robot
```

## Reference
```
@article{yu2024robustify,
  title     = {Robustifying Human-Robot Collaboration through a Hierarchical and Multimodal Framework},
  author    = {Yu, Peiqi and Abuduweili Abulikemu and Liu, Ruixuan and Liu, Changliu},
  journal={arXiv preprint arXiv:xxx},
  year={2024}
}
```



