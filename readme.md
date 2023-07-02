# Experiment
________________________________
- Using Deep Q-learning, the RL agent learned how to balance the pole by applying forces in the left and right direction on the cart
  - ![simutation_v_2_0_5.gif](./img_n_video/simutation_v_2_0_5.gif)
- With the help of wandb, monitor the episode duration instantly while training. The agent play well about 400 episodes.
  - ![Optional Text](./img_n_video/episode_duraions_curve.png)

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

# File
1. train_model.py: train the model
2. simulate.py: simulate the game using the trained model
3. cfg.py: the configuration of model and dataset processing
