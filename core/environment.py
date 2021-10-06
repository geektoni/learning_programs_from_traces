from abc import ABC


class Environment(ABC):

    def __init__(self):
        pass

    def start_task(self, task_index):
        return 0, 0

    def get_state(self):
        pass

    def end_task(self):
        pass