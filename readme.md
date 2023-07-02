# File
1. train_model.py: train the model
2. simulate.py: simulate the game using the trained model
3. cfg.py: the configuration of model and dataset processing

# Experiment
________________________________

![alt text](https://github.com/Auyin111/rl_game/edit/master/img_n_video/episode_duraions_curve.png)


# Principle of Deep Q-learning
___________________________________

  - uses a deep neural network to approximate the different Q-value for each possible action at a state (value-function estimation)
  - has two phases:
    1. sampling: perform actions and store the observed expectations tuples in a replay memory
    2. Training: Select the small batch of tuple randomly and learn from it using a gradient descent update step
  - pseudocode
    1. Initialize replay memory D to capacity N
    2. Initialize action-value function Q with random weight \theta
    3. Initialize <mark>target</mark> action-value function \hat{Q} with weights \theta^_ = \theta
    4. For episode = 1, M do
       1. Initialize sequence s_1 = {x_1} and preprocessed sequence \phi_1 = \phi(s_1)
       2. For t = 1, T do
          - <mark>(5 Sampling steps)</mark>
          1. With probability \epsilon select a random action a_t
          2. otherwise select <img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white} a_t=\arg\max_a Q(\phi(s_t), a;\theta)" /><br>
          3. Execute action a_t in emulator (模擬器) and observe reward r_t and image x_{t+1}
          4. Set <img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white} s_{t+1}=s_t,a_t,x_{t+1} \;and\;preprocess\; \phi_{t+1}=\phi(s_{t+1})" /><br>
          5. Store transition (<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white} (\phi_t,a_t,r_t,\phi_{t+1})\;in\;D" /><br>)
          - <mark>(5 Training steps)</mark>
          1. Sample random minibatch of transition (<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white} (\phi_j,a_j,r_j,\phi_{j+1})\;from\;D" /><br> 
          - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}  Set\;y_j=\left\{\begin{matrix} r_j & \text{if episdeo terminates at step j+1} \\ r_j+\gamma\max_{a'}\hat{Q}(\phi_{j+1},a';\theta^- & otherwise  \end{matrix}\right." /><br>
          - Perform a gradient decent step on <img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}  (y_j-Q(\phi_j,a_j;\theta))^2 \text{ with respect to the network parameter } \theta"  /><br>
          - Every C steps reset \hat(Q) = Q
  - training might suffer from instability. Mainly because of combining a non-linear Q-value function (NN) and bootstrapping (when we update targets <mark>with existing estimates and not an actual</mark> complete return)
    - Solution
      1. Experience Replay
         1. allows us to learn from individual experience multiple times (avoid forgetting previous experiences)
         2. remove correlation in the observation sequences and avoid action values from oscillating or diverging catastrophically 
         
