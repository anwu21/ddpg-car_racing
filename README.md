This repository contains code and sample results for running the [Deep Deterministic Policy Gradient (DDPG) algorithm](https://arxiv.org/abs/1509.02971) in the [OpenAI Gym](https://gym.openai.com/) Car-Racing environment. DDPG is an actor-critic model-free RL algorithm that operates over continuous action spaces. Like DQN, DDPG uses a replay buffer to stabilize training. DDPG also makes use of target networks, but with soft updates using polyak averaging rather than hard updates. This way, the target values change slowly and stability improves greatly during training.

The DDPG algorithm (from the original paper) is below:

![ddpg_algorithm](/figures/ddpg_algorithm.png)

The OpenAI Gym Car-Racing environment consists of a car going around a racetrack from a birdseye view. The simulator allows for continuous control of the car and learning directly from pixels. The state is the top down view. The agent is rewarded by the number of tiles of the track it visits and penalized by the number of frames (i.e. time) it takes to drive around the track. During training, every episode consists of a randomly generated track. Below is a plot of rewards accumulated over the training period. The green dots in the background reflect the reward for the episode and the black line is a (moving) average of the rewards.

![ddpg_rewards](/figures/ddpg_rewards.png)

And here are some sample results of agents as they learn to drive around the track. 

Course 1:

![course1_run0](/output/course1/run_0.gif)

Episode 750


![course1_run1](/output/course1/run_1.gif)

Episode 1500


![course1_run2](/output/course1/run_2.gif)

Episode 2250


![course1_run3](/output/course1/run_3.gif)

Episode 3000



Course 2:

![course2_run0](/output/course2/run_0.gif)

Episode 750


![course2_run1](/output/course2/run_1.gif)

Episode 1500


![course2_run2](/output/course2/run_2.gif)

Episode 2250


![course2_run3](/output/course2/run_3.gif)

Episode 3000



# Requirements

The code has been tested on Ubuntu 16.04 using python 3.5, [PyTorch](pytorch.org) version 0.3.0 with a Titan X GPU.


# Setup

1. Download the code ```git clone https://github.com/anwu21/ddpg-car_racing.git```

2. To train, type: python3 main.py --mode train

3. To test, type: python3 main.py --mode test
   - If you have more than 1 model in the "output" directory, you will need to enter the model number. For example, if you would like to test model #5, you will need to type: python3 main.py --mode test --model_path 5
