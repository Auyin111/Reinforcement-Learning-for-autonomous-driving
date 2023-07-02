class Cfg:
    """
    if test_mode == True, only use low computation setting to debug the code
    """

    def __init__(self, test_mode, dir_data):

        self.test_mode = test_mode
        self.dir_data = dir_data

        # Define
        self._define_hyper()
        self._define_constant()
        self._define_wandb()

    def _define_hyper(self):
        """define hyperparameters"""
        self.HID_SIZE = 20 if self.test_mode else 256
        self.LR = 1e-4

    def _define_constant(self):
        # discount factor
        self.GAMMA = 0.99

        # number of transitions sampled from the replay buffer
        self.BATCH_SIZE = 10 if self.test_mode else 256

        self.NUM_EPISODES = 40 if self.test_mode else 600
        # memory capacity
        self.MEMORY_CAP = 10 if self.test_mode else 10000

        ### epsilon
        # starting value
        self.EPS_START = 0.9
        # final value
        self.EPS_END = 0.05
        # rate of expoential decay of epsilon, higher means a slower decay
        self.EPS_DECAY = 1000

        # update rate of the target network
        self.TAU = 0.005

        # display
        self.EPISODE_PRINT = 20

    def _define_wandb(self):
        self.project = 'rl_CartPole'
        self.user = 'Kyle'
        self.version = 'v_2_0_5'
        self.model = 'DQN'
        self.entity = 'kaggle_winner'
