import numpy as np

from . import yag_screen
from . import emittance_calculation
import logging
logger = logging.getLogger(__name__)

class Emittance(yag_screen.YAGScreen):
    def __init__(self, interface, px_scale, slit_sep_m, R12, R11=1.0,
                 target_charge=-1, charge_deviation=0.1, average_measurements=False,
                 image_directory='emittance', n_samples=1, n_required_blobs=1):

        """

        Parameters
        ----------
        interface : AWAInterface object
            Low level interfacing with the accelerator

        px_scale : float
            Pixel scaling in m / pixel

        slit_sep_m : float
            Seperation between adjcent slits in meters

        R12 : float
            R12 matrix element, usually the drift length in meters

        R11 : float, default=1.0
            R11 matrix element, no units

        target_charge : float, default=-1
            Target charge for measurement in nC. Negative numbers prevent charge windowing

        charge_deviation : float, default=0.1
            Fractional charge deviation allowed

        average_measurements : bool, default=False
            Flag that determines if multiple samples are averaged to reduce the number of data points.

        image_directory : str, default='pics'
            Directory to store measurements

        n_samples : int, default=1
            Number of samples to take
        """

        self.px_scale = px_scale
        self.slit_sep_m = slit_sep_m
        self.R12 = R12
        self.R11 = R11
        self.average_measurments = average_measurements

        self.n_required_blobs = n_required_blobs

        super(Emittance, self).__init__(interface, target_charge, charge_deviation,
                                        image_directory, n_samples)

    def measure_emittance(self, **kwargs):
        # do screen measurements
        screen_measurements = self.measure_screen(**kwargs)
        n_measurements = len(screen_measurements['processed_images'])

        valid_measurements = np.all((#screen_measurements['image_check'],
                                     screen_measurements['total_intensity'] > 250,
                                     #screen_measurements['rms_y'] < 300,
                                     screen_measurements['n_blobs'] >= self.n_required_blobs), axis=0)

        # for each valid measurement index calculate the emittance
        emittances = []
        for ii in range(n_measurements):
            if valid_measurements[ii]:
                logger.info('doing emittance calculations')
                emittances += [emittance_calculation.calculate_emittance(
                    screen_measurements['processed_images'][ii],
                    self.px_scale,
                    self.slit_sep_m,
                    self.R12)]

            else:
                emittances += [np.NaN]

        emittances = np.array(emittances)
        screen_measurements.update({'EMITY': emittances})

        # if we need to, get averaged results
        if self.average_measurments:
            avg_keys = ['EMITY', 'rms_x', 'rms_y', 'CX', 'CY', 'n_blobs', 'FWHMX', 'FWHMY', 'centroid_offset', 'total_intensity']
            for key in avg_keys:
                screen_measurements[key] = np.nanmean(screen_measurements[key])
                screen_measurements[key + '_std'] = np.nanstd(screen_measurements[key])

        return screen_measurements
