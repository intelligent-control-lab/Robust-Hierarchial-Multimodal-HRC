# Robustifying Long-term Human-Robot Collaboration through a Hierarchical and Multimodal Framework
![Architecture](/assets/images/architecture.png)

## Abstract
Long-term Human-Robot Collaboration (HRC) is crucial for developing flexible manufacturing systems and for integrating companion robots into daily human environments over extended periods. However, sustaining such collaborations requires overcoming challenges such as accurately understanding human intentions, maintaining robustness in noisy and dynamic environments, and adapting to diverse user behaviors.
This paper presents a novel **multimodal** and **hierarchical** framework to address these challenges, facilitating efficient and robust long-term HRC. In particular, the proposed multimodal framework integrates visual observations with speech commands, which enables intuitive, natural, and flexible interactions between humans and robots. Additionally, our hierarchical approach for human detection and intention prediction significantly enhances the system's robustness, allowing robots to better understand human behaviors. The proactive understanding enables robots to take timely and appropriate actions based on predicted human intentions. 
We deploy the proposed multimodal hierarchical framework to the KINOVA GEN3 robot and conduct extensive user studies on real-world long-term HRC experiments. 
***


This is the official code repo for the paper [Robustifying Human-Robot Collaboration through a Hierarchical and Multimodal Framework](https://arxiv.org/abs/2411.15711). The video demo can be found at [https://www.youtube.com/watch?v=2kkANN9ueVY](https://www.youtube.com/watch?v=2kkANN9ueVY).

## Youtube Video:
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
  title     = {Robustifying Long-term Human-Robot Collaboration through a Hierarchical and Multimodal Framework,
  author    = {Peiqi Yu  and Abulikemu Abuduweili and Ruixuan Liu and Changliu Liu },
  journal={arXiv preprint arXiv:2411.15711},
  year={2024}
}
```



