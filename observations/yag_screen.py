import logging

import numpy as np

from . import image_processing
import h5py as h5py
import time


def apply_roi(image, roi_coords):
    """
    get portion of image that is inside region of interest rectangle

    Arguments
    ---------
    image : ndarray, size (N x M)
        Image array

    roi_coords : ndarray, size (2,2)
        Region of interest coordinates in the form
        ((x1,y1),(x2,y2)) where (x1,y1) is the lower left corner
        and (x2,y2) is the upper right corner

    """
    return image.T[roi_coords[0, 0]:roi_coords[1, 0], roi_coords[0, 1]:roi_coords[1, 1]].T


class YAGScreen:
    def __init__(self, interface, target_charge=-1, charge_deviation=0.1,
                 image_directory='pics', n_samples=1, save_images=True, average_measurements=False):

        """
        YAGScreen measurement class

        Arguments
        ---------
        target_charge : float, optional
            Target charge for valid observation in nC, if negative ignore.
            Default: -1 (ignore)

        charge_deviation : float, optional
            Fractional deviation from target charge on ICT1 allowed for valid
            observation. Default: 0.1

        image_directory : str, optional
            Location to save image data. Default: None

        n_samples : int, optional
            Number of samples to take. Default: 1


        """
        self.logger = logging.getLogger(__name__)

        self.n_samples = n_samples
        self.target_charge = target_charge
        self.charge_deviation = charge_deviation

        self.save_image_flag = save_images
        self.image_directory = image_directory

        self.interface = interface
        self.average_measurements = average_measurements

        self.output_keys = ['ellipses', 'processed_images', 'rms_x',
                            'rms_y', 'n_blobs', 'image_check',
                            'centroid_offset', 'total_intensity']

    def save_images(self, data_dict):
        self.logger.debug(f'saving image data to {self.image_directory}')
        fname = f'{self.image_directory}/img_{time.time()}.h5'
        with h5py.File(fname, 'w') as f:
            for name, item in data_dict.items():
                if 'images' in name:
                    f.create_dataset(name, data=item)
                elif name=='ellipses':
                    pass
                else:
                    if item is not None:
                        f['/'].attrs[name] = item

    def _get_images(self, **kwargs):
        """
        Get data from the controller_interface and check its validity

        Check the following:
        - that a reasonable roi has been specified (bigger than 2 px in each direction
        - that there is a beam in the roi/the beam is not cut off in the roi

        NOTE: we consider charge in the controller_interface so that measurements can be repeated quickly

        If one of these conditions are not met then set 'IMGF' to 0 and return Nans for the
        beamsize/location data.

        """
        raw_outputs = self.interface.get_data(self.target_charge,
                                              self.charge_deviation,
                                              n_samples=self.n_samples)

        # apply roi to images
        roi_images = []
        for i in range(self.n_samples):
            roi_images += [apply_roi(raw_outputs['raw_images'][i], raw_outputs['ROI'])]

        # process and identify blobs in image
        outputs = {}
        for ele in self.output_keys:
            outputs[ele] = []

        for i in range(len(roi_images)):
            processed_image_data = image_processing.process_and_fit(roi_images[i],
                                                                    **kwargs)

            for ele in self.output_keys:
                if ele == 'image_check':
                    outputs[ele] += [image_processing.check_image(processed_image_data['binary_image'],
                                                                  processed_image_data['post_processed_image'])]
                elif ele == 'processed_images':
                    outputs[ele] += [processed_image_data['post_processed_image']]
                else:
                    outputs[ele] += [processed_image_data[ele]]

        for ele in self.output_keys:
            outputs[ele] = np.array(outputs[ele])

        # add in raw data
        outputs.update(raw_outputs)

        # if we need to, get averaged results
        if self.average_measurements:
            avg_keys = ['rms_x', 'rms_y', 'CX', 'CY', 'n_blobs', 'FWHMX', 'FWHMY', 'centroid_offset', 'total_intensity']
            for key in avg_keys:
                out = outputs[key].copy()
                outputs[key] = np.nanmean(out)
                outputs[key + '_std'] = np.nanstd(out)


        return outputs

    def measure_screen(self, **kwargs):
        """
        Do screen measurement using controller_interface

        """
        outputs = self._get_images(**kwargs)

        if self.save_image_flag:
            self.save_images(outputs)
        return outputs
