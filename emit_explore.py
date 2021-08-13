import time
import awa_interface
from observations.emittance import Emittance
from xopt.bayesian.algorithms import bayesian_exploration

import logging

logging.basicConfig(level=logging.DEBUG)


class Evaluator:
    def __init__(self, interface):
        self.interface = interface

    def evaluate_emittance(self, inputs):
        self.interface.set_parameters(inputs)

        time.sleep(1.0)

        observation = Emittance(self.interface,
                                50e-3 / 1000,
                                0.002,
                                1.27,
                                n_samples=2,
                                average_measurements=True)

        results = observation.measure_emittance()

        return results


VOCS = {
    'name': 'AWA_Exp',

    'variables': {
        'Drive:Ctrl:DS1': [6.05, 9.07],  # Focusing Solenoid
        'Drive:Ctrl:DS3': [0.0, 2.5],  # Matching Solenoid
        'SMCtrl:AWA:abs03': [-640, 640],  # Linac Phase
        'AWA:Bira3Ctrl:Ch03': [0.0, 0.5],  # DQ4
        'AWA:Bira3Ctrl:Ch04': [0.0, 0.5],  # DQ5
        'AWA:Bira3Ctrl:Ch05': [-0.25, 0.0],  # DQ6

    },

    'objectives': {
        'EMITY': None,
    },

    'constraints': {
        'rms_x': ['LESS_THAN', 400],
        'rms_y': ['LESS_THAN', 400],
        'centroid_offset': ['LESS_THAN', 100],
        'n_blobs': ['GREATER_THAN', 3]

    }

}

awa_interface = awa_interface.AWAInterface(testing=True)
evaluator = Evaluator(awa_interface)
opt_results = bayesian_exploration(VOCS,
                                   evaluator.evaluate_emittance,
                                   n_steps=0,
                                   n_initial_samples=1)
