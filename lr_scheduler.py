import torch
import numpy as np


class CustomLRScheduler:

    def __init__(self,
                 optimizer,
                 initial_value: float or int,
                 decay_type: str,
                 decay_rate: float = None,
                 decay_steps: int = None,
                 min_value: float or int = None,
                 value_name: str = None,
                 verbose: bool = False,
                 ):

        # Keeps track of and decays a given value in a given way

        self.optimizer = optimizer

        self.value = float(initial_value)

        self.decay_type = decay_type
        self.decay_rate = float(decay_rate) if decay_rate is not None else 0.9
        self.decay_steps = float(decay_steps) if decay_steps is not None else None
        self.min_value = float(min_value) if min_value is not None else None
        self.value_name = value_name if value_name is not None else 'Learning Rate to be scheduled'
        self.verbose = verbose

        self._step = None  # Internal step function decrementing self.value each time that step()-method is invoked

        self.init_step_function()


    def init_step_function(self):
        # Assign function that will decay the value to be decayed whenever step() method is called

        if self.min_value is None:
            self.min_value = 0.
        elif isinstance(self.min_value, int):
            self.min_value = float(self.min_value)

        if self.decay_type.lower() == 'constant':
            # Handle cases where value is not supposed to decay
            self._step = lambda: self.value

        elif self.decay_type.lower() == 'linear':
            # Handle cases where decay type is 'linear'

            if self.decay_steps is not None:
                # We know over which time span to linearly anneal the initial value to the min value
                # Thus, compute the decay rate for linear decay
                self.decay_rate = (self.value - self.min_value) / self.decay_steps

            # After we know the decay rate (which is either provided or has been computed above), next compute function
            # decaying the value each scheduler-step
            self._step = lambda: float(np.nanmax([self.min_value, self.value - self.decay_rate]))

        elif self.decay_type.lower() == 'exponential':
            # Handle cases where decay type is 'exponential'
            self._step = lambda: float(np.nanmax([self.min_value, self.value * self.decay_rate]))

        else:
            raise NotImplementedError('Scheduler can only handle linear or exponential decay.')


    def step(self):
        # Decrease the scheduled value by one quantity

        if self.verbose:
            print("Going to decrease", self.value_name, self.decay_type + "ly by factor {:.5f}".format(self.decay_rate),
                  "to {:.10f}".format(self._step()) + ".")

        # Update current learning rate
        self.value = self._step()

        # Update learning rate to previously computed new value
        for g in self.optimizer.param_groups:
            g['lr'] = self.value