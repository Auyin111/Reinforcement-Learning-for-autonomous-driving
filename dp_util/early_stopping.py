import torch


class EarlyStopper:

    def __init__(self, dir_model,
                 patience=10, smaller_is_better=True, delta=0.,
                 sf=4,
                 verbose=True, num_epoch_display=5, num_tried_display=5

                 ):
        """
        :param smaller_is_better: if True, the smaller the observed value is better
        :parma delta: minimum change to qualify as an improvement
        """

        self.dir_model = dir_model

        self.patience = patience
        self.smaller_is_better = smaller_is_better
        self.delta = delta

        self.sf = sf

        self.verbose = verbose
        self.num_epoch_display = num_epoch_display
        self.num_tried_display = num_tried_display

        self.best_value = None
        self.epoch = 1
        self.num_tried = 1
        self.early_stop = False

    def __call__(self, value, model):

        # if before +1 already == patience, raise error
        if self.num_tried == self.patience:
            raise Exception("already early stopped, don't call the method again")

        if self.best_value is None:

            self.best_value = value
            if self.verbose:
                print(f'the value is started at {value:.{self.sf}f}')
        else:
            if self.smaller_is_better:
                improved = value < (self.best_value - self.delta)
            else:
                improved = value > (self.best_value + self.delta)

            if improved:
                self.num_tried = 1
                self.epoch += 1
                if self.verbose:
                    if self.epoch % self.num_epoch_display == 0:
                        msg = f'the value is improved from {self.best_value:.{self.sf}f} to {value:.{self.sf}f} (best_epoch: {self.epoch})'
                        print(msg)
                self.best_value = value
                self.save_cp(model)
            else:
                self.num_tried += 1
                if self.verbose:
                    if self.num_tried % self.num_tried_display == 0:
                        msg = f'the value is remained {self.best_value:.{self.sf}f}, the new value is {value:.{self.sf}f} ({self.num_tried}/{self.patience})'

        if self.num_tried == self.patience:
            self.early_stop = True
            print(f'\nEarly stop at best_epoch = {self.epoch} and best_value = {self.best_value:.{self.sf}f}')

    def save_cp(self, model):
        torch.save(model.state_dict(), self.dir_model)
