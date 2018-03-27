import tensorflow as tf

class config():
    def __init__(self):
        self.mov_num   = 11
        self.beta      = 0.8

        self.learning_rate = 1e-3
        self.lr_decay      = 0.1
        self.decay_steps   = 1e4
        self.steps         = int(1e5)