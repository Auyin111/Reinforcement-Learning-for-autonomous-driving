class OnTrackClsCfg:
    """
    if test_mode == True, only use low computation setting to debug the code
    """

    # in test mode, only use less computation setting to debug the code

    def __init__(self, test_mode, dir_data):
        self.test_mode = test_mode
        self.dir_data = dir_data

        # Define
        self._define_hyper()
        self._define_constant()
        self._define_wandb()

    def _define_hyper(self):
        """define hyperparameters"""
        self.LR = 4e-4

    def _define_constant(self):
        self.SF = 5
        self.BATCH_SIZE = 10 if self.test_mode else 32

    def _define_wandb(self):
        self.project = 'on_track_classifier'
        self.user = 'Kyle'
        self.version = 'v_2_0_5'
        self.model = 'CNN'
        self.entity = 'kaggle_winner'
        self.md_name = f'{self.project}_{self.version}_test.pth' if self.test_mode else f'{self.project}_{self.version}.pth'
