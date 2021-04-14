
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: https://video.udacity-data.com/topher/2018/July/5b48f845_unknown/unknown.png "Benchmark"


# Project Continuous Control
Assignment for Udacity Deep Reinforcement Learning Nanodegree  


## Introduction

We want to train an agent to act in a continuous space.

In the environment we consider here a double-jointed arm can move continuously to target locations. The goal of our agent is to maintain its position at the target location for as many time steps as possible.


### Environment

For this project, we work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
The environment involves controlling a double-jointed arm to reach target locations.

![Trained Agent][image1]

A *reward* of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The *observation* space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

Each *action* is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we have two separate versions of the Unity environment:

- The first version contains a single agent.  

- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  


### Problem to Solve
The goal of our agent is to maintain its position at the target location for as many time steps as possible.

The task is episodic and in order to solve the environment the agent must get an average score of +30 over 100 consecutive episodes, where a reward of +0.1 is gained for each step at target location.

An example of satifactory solution is 

![Benchmark][image3]


#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 




## Installation

### Python Dependencies

- The software requires to install Python (3.6.1 or higher). We advocate to create a new environment with Python 3.6
     
     - Linux or Mac:
  
    ```sh
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    
    - Windows:
  
    ```sh
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

- Clone the repository, and navigate to the python/ folder. Then, install several dependencies.

    ```sh
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install .
    ```
- Create and activate IPython kernel for the drlnd environment.

    ```sh
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    In the jupyter notebook instance the kernel is activated from the dropdown menu _Kernel_



### Unity Packages
Besides the Python ML library `PyTorch` you will need to install the Unity Packages and Environments plus the relevant Python Packages following the instructions in [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

The ML-Agents Toolkit contains several components:

- Unity package `com.unity.ml-agents` contains the
  Unity C# SDK that will be integrated into your Unity project.  This package contains
  a sample to help you get started with ML-Agents.
  
- Unity package `com.unity.ml-agents.extensions` contains experimental C#/Unity components that are not yet ready to be part
  of the base `com.unity.ml-agents` package. `com.unity.ml-agents.extensions`
  has a direct dependency on `com.unity.ml-agents`.
  
- Three Python packages:
    - `mlagents` contains the machine learning algorithms that
      enables you to train behaviours in your Unity scene. Most users of ML-Agents
      will only need to directly install `mlagents`.
    - `mlagents_envs` contains a Python API to interact with
      a Unity scene. It is a foundational layer that facilitates data messaging
      between Unity scene and the Python machine learning algorithms.
      Consequently, `mlagents` depends on `mlagents_envs`.
    - `gym_unity` provides a Python-wrapper for your Unity scene
      that supports the OpenAI Gym interface.
      
### Environment

- Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - _Version 1: One (1) Agent_
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - _Version 2: Twenty (20) Agents_
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

- Place the file in the working directoty folder and unzip (or decompress) the file. 
      

### Instructions

You can either follow the steps in the python notebook `Continuous_Control.ipynb` or run it locally from `Continuous_Control.py`.

### Directory Structure

1. Main Python Notebook `Continuous_Control.ipynb`

2. Main Python Code `Continuous_Control.py`

3. Python module `ddpg_agent.py` defines class Agent that learns by interacting with environment.

4. Python module `ddpg_model.py` defines Actor and Critic deep neural networks. 

5. Python module `ddpg_interact.py` defines how the agent interacts with the environment either learning or following best policy. 

6. The directory `./Reacher_LinuxNoVis_20/Reacher.x86_64` contains the Unity compiled program for the environment with no visualization.

7. The PyTorch file `Trained_Agent.pth` is the trained model with weights of the Q Network.  


### GPU
If Cuda library available PyTorch will automatically run on GPU otherwise on cpu.

## License
MIT License

Copyright (c) [2021] [A.M.C.S.]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

