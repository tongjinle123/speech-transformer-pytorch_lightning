from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    """
    not usable for pytorch lightning
    """
    def __init__(self, optimizer, warm_up_steps, model_size, factor, last_epoch=-1):
        self.warm_up_steps = warm_up_steps
        self.factor = factor
        self.model_size = model_size
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [
            self.factor * (
                    (self.model_size ** -0.5) * min(self._step_count ** -0.5, self._step_count * (self.warm_up_steps ** -1.5))
            )
            for base_lr in self.base_lrs
        ]
        return lrs
