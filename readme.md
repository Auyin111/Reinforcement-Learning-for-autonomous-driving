# Experiment 1
________________________________
- Using Deep Q-learning, the RL agent learned how to <mark>drive the car</mark> by turn left, turn right, gas, break and do nothing
- Epochs 100:
  - ![driving.gif](./img_n_video/driving_demo_epochs_100.gif)
- Epochs 250:
  - ![driving.gif](./img_n_video/driving_demo_epochs_250.gif)
- Key trick of training this model
  1. A <mark>large amount of epochs</mark> is required to train an agent to drive a car
  2. Clip the image to remove useless information to reduce the noise and memory usage. It can also increase the capacity of replay buffer
  3. As the image is zooming in the first 50 steps and it wil affect learning, do not interact with env
  4. As the car able to gain the reward if it drives near to the glass, I used <mark>CNN to build another classifier model determine whether off track</mark>. if off track, penalise
     - ![driving.gif](./img_n_video/on_track_cls_avg_loss_curve.png)
     - ![driving.gif](./img_n_video/on_track_cls_perf.png)
  5. The output of 'state' from the env is just a static image. The agent <mark>can not determine the velocity of car so the agent is impossible to make a good decision</mark>. To solve this problem, I use both the 'state' and 'prev_state' as input  
  6. During training, the car is quite inactive and tend to stay in the same point. 
     1. Add more reward if it has positive reward
     2. In the early stage, increase the chance of using gas and reduce the chance of using break
  7. The training time is very long, the capability of further train a previous saved model is very important
  
# Experiment 2
________________________________
- Using Deep Q-learning, the RL agent learned how to <mark>balance the pole</mark> by applying forces in the left and right direction on the cart
  - ![simutation_v_2_0_5.gif](./img_n_video/simutation_v_2_0_5.gif)
- With the help of wandb, monitor the episode duration instantly while training. The agent play well about 400 episodes.
  - ![Optional Text](./img_n_video/episode_duraions_curve.png)

### File
1. train_model.py: train the model
2. simulate.py: simulate the game using the trained model
3. cfg.py: the configuration of model and dataset processing

# Principle of Deep Q-learning
___________________________________

  - uses a deep neural network to approximate the different Q-value for each possible action at a state (value-function estimation)
  - has two phases:
    1. sampling: perform actions and store the observed expectations tuples in a replay memory
    2. Training: Select the small batch of tuple randomly and learn from it using a gradient descent update step
![Optional Text](./img_n_video/DQN_psaudocode.png)
  - training might suffer from instability. Mainly because of combining a non-linear Q-value function (NN) and bootstrapping (when we update targets <mark>with existing estimates and not an actual</mark> complete return)
    - Solution
      1. Experience Replay
         1. allows us to learn from individual experience multiple times (avoid forgetting previous experiences)
         2. remove correlation in the observation sequences and avoid action values from oscillating or diverging catastrophically 

# Principle of Double DQN
-------------------------------

- One of the problems of the DQN algorithm is that is overestimates the true rewards
- To fix this, DDQN suggest using a simple trick:
  - decoupling the action selection from the action evulation

![Optional Text](./img_n_video/double_dqn_flowchart.png)
![Optional Text](./img_n_video/double_dqn_formula.png)

# Principle of Dueling DQN
-------------------------------

- splits the Q-values in two different parts, the value function V(s) and the advantage function A(s,a)
- V(s) tell us how much reward we will collect from state s
- A(s,a) tells us how much better one action is compared to the other actions
- sometimes it is unnecessary to know the exact value of each action, so just learning the state-value function can be enough in some cases 

![Optional Text](./img_n_video/dueling_dqn_architecture.png)
![Optional Text](./img_n_video/dueling_dqn_formula.png)


