import time
import awa_interface
from observations.emittance import Emittance
from xopt.bayesian_exploration import bayesian_exploration


def evaluate_emittance(inputs, interface):
    interface.set_parameters(inputs)

    time.sleep(1.0)

    observation = Emittance(interface,
                            50e-3 / 700,
                            0.002,
                            0.87,
                            n_samples=2)

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
        'FWHMX': ['LESS_THAN', 400],
        'FWHMY': ['LESS_THAN', 400],
        'CX_DEV': ['LESS_THAN', 100],
        'CY_DEV': ['LESS_THAN', 100],
        'N_BEAMLETS': ['GREATER_THAN', 3]

    }

}

awa_interface = awa_interface.AWAInterface(testing=True)
opt_results = bayesian_exploration(VOCS,
                                   evaluate_emittance,
                                   n_steps=1,
                                   n_initial_samples=1,
                                   eval_args=[awa_interface])
