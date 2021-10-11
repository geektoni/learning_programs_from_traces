import numpy

class Policy():

    def __init__(self):
        pass

    def init_tensors(self):
        return numpy.zeros(1), numpy.zeros(1)

    def forward_once(self, observation, program_index, h, c):
        pass