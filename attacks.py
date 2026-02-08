# import torch

class Attack:
    def __init__(self, name):
        self.name = name

    def manipulate_update(self, update):
        return update


class ByzantineFlip(Attack):
    def manipulate_update(self, update):
        return -update
