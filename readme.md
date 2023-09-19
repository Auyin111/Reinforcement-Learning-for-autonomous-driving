# 2D Reinforcement Learning Autonomous car (gym package: CarRacing-v2)
- Using Double Dueling Deep Q-learning (RL) and Physics knowledge to train the RL agent to drive the car with the action of turning left, turning right, gas, breaking, and doing nothing

## Trick 1: Off-track penalty
- In v_3_1_3, the CNN off-track classifier gives a big penalty if the car is off-track. <mark>It can prevent the car driving on the grass **unnecessarily**</mark>

Info| v_3_1_2: without off track penalty          |  v_3_1_3: with off track penalty
:-------------------------:|:-------------------------:|:-------------------------:
Demo 1 | ![driving.gif](./img_n_video/v_3_1_2_normal.gif) |  ![driving.gif](./img_n_video/v_3_1_3_normal_2.gif)
Demo 2 | ![driving.gif](./img_n_video/v_3_1_2_normal_2.gif) |  ![driving.gif](./img_n_video/v_3_1_3_normal.gif)
Training time | Relative short| Relative long
Task difficulty | Low | High

### Learning curve:
  - As the task difficulty of v_3_1_2 is easier, it uses the lesser time to gain 1000 round rewards
  - ![driving.gif](./img_n_video/DDDQN_learning_curve.png)

### CNN off-track classifier:
  - use 1500 human-labeled images where half of them are 'on track' and half of them are 'off track' to train, valid and test the classifier
  - ![driving.gif](./img_n_video/on_track_cls_perf.png)
  - ![driving.gif](./img_n_video/on_track_cls_avg_loss_curve.png)

## Trick 2: Consider "Velocity" when driving:
  - The output of 'state' from the env is just a static image. If the agent does not know the current velocity, it ***_can not determine how much centripetal force is required to make a turn_***
  - ![driving.gif](./img_n_video/car_turning_top.jpg)
  - To solve this problem, need to use <mark>both the 'state' and 'prev_state'</mark> as input
    - because using the original coordinate and final coordinate can derive the displacement and <mark>velocity</mark>
    - **Velocity formula**
        ```math
        v = \frac{\Delta s}{\Delta t} = \frac{x_f-x_i}{t_f-t_i}
        ```
        where v is the velocity, s is the displacement, t is the time and x is the coordinates

## Trick 3: State processing
  1. As the image is zooming in the first 50 steps so the state (vision) is abnormal and confusing, do not interact with env during that 50 steps
  2. The state (vision) generated from gym includes useless information such as the cumulative reward, current power, and baking power. Clip that useless information can reduce the noise and increase the capacity of the replay buffer   

## Trick 4: Speed up the car and training process
  - As it is racing, the higher the velocity is better
    1. Give a reward if the speed is higher and penalize if the speed is too low
    2. Increased the chance of using gas and reduce the chance of using break, it will realize faster is better in a shorter time 


## Current limitation
  - sometimes the car will off-track if the car comes to a <mark>tight turn with a high velocity</mark>
    - it is a Physics limitation
      - if the car is too fast, most of the displacement is due to the original velocity but not operation adjustment. Therefore, it is impossible to turn it well
        - **Velocity, displacement, and acceleration formula**
      ```math
      \displaylines{v = \frac{\Delta s}{\Delta t} = \frac{x_f-x_i}{t_f-t_i} \\ s =  ut + \frac{1}{2}at^2 = \text{displacement is due to the original velocity + displacement is due to the operation adjustment} \\ a = \frac{\Delta v}{\Delta t}}
      
      ```
      where v is the velocity, s is the displacement, t is the time, x is the coordinates and a is the acceleration
    - ![driving.gif](./img_n_video/v_3_1_3_too_fast.gif)
  1. the best method to fix this issue is to allow the ***_agent to look further (zoom out the image)_***
     1. if the agent views the tight turn early, it can <mark>reduce the car speed early</mark> and it will reduce the displacement due to the original velocity
     2. if the agent views the tight turn early, it can <mark>shift the car to the outermost point early</mark> in the road to ***_reduce the required centripetal force_***
     - ![driving.gif](./img_n_video/v_3_1_3_prepare_turning.gif)
  3. ***_give a penalty if the car is too fast_*** to reduce the displacement due to the original velocity. but it will limit the car's velocity.


### File
1. train_on_track_cls.py: train, valid and test the on-track classifier model
2. train_rl_racing.py: train and test the RL car racing agent
3. simulate_rl_racing.py: simulate the car driving in a random environment

# Principle of Deep Q-learning
  - uses a deep neural network to approximate the different Q-value for each possible action at a state (value-function estimation)
  - Formula:
    ```math
    \displaylines{Q(S_t,At)\leftarrow Q(S_t,A_t) +\alpha[R_{t+1}+\gamma max_aQ(S_{t+1},a)-Q(S_t,A_t)]
    \\\text{where } R_{t+1}+\gamma max_aQ(S_{t+1},a) \text{ is TD Target,}
    \\R_{t+1}+\gamma max_aQ(S_{t+1},a)-Q(S_t,A_t) \text{ is TD Error}
    }
    ```
  - has two phases:
    1. sampling: perform actions and store the observed expectations tuples in a replay memory
    2. Training: Select the small batch of tuple randomly and learn from it using a gradient descent update step
  ![Optional Text](./img_n_video/DQN_psaudocode.png)
  - training might suffer from instability. Mainly because of combining a non-linear Q-value function (NN) and bootstrapping (***_when we update targets <mark>with existing estimates and not an actual</mark> complete return_***)
  - Solution to stabilize the training
    1. Experience Replay
       1. allows us to learn from individual experiences multiple times
       2. remove correlation in the observation sequences and avoid action values from oscillating or diverging catastrophically
    2. Fixed Q-Target to stabilize the training
       -  if both Q-values and the target values shift at every step of training, this can ***_lead to significant oscillation in training_***
       1. use another network with fixed parameters for estimating the TD target
       2. copy the parameters from Deep Q-network every C steps to update the target
    3. Double DQN 
       - As using the same values both to select and to evaluate an action, DQN algorithm ***_overestimates the true rewards_***
       - To fix this, DDQN suggests using a simple trick:
         - ***_decoupling the action selection from the action evaluation_***
           1. use DQN network to select the best action to take for the next state (the action with highest Q-value)
           2. Use Target network to calculate the target Q-value of taking that action at the next state
      ![Optional Text](./img_n_video/double_dqn_flowchart.png)
      ![Optional Text](./img_n_video/double_dqn_formula.png)
    4. Dueling DQN
       - splits the Q-values in two different parts, the value function V(s) and the advantage function A(s,a)
         - V(s) tell us how much reward we will collect from state s
         - A(s,a) tells us how much better one action is compared to the other actions
         - at the end, combines both parts into a single output
       - sometimes it is unnecessary to know the exact value of each action, so ***_just learning the state-value function can be enough in some cases_***
      
       ![Optional Text](./img_n_video/dueling_dqn_architecture.png)
       ![Optional Text](./img_n_video/dueling_dqn_formula.png)
