class LRSchedule(object):
    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def constant(x, lr_init=0.1, epochs=1):
        return lr_init
