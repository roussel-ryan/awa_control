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
                 image_directory='pics', n_samples=1):

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

        self.save_image_flag = False
        self.image_directory = image_directory

        self.interface = interface

    def save_images(self, data_dict):
        # determine if we are saving images
        if self.image_directory is None:
            self.save_image_flag = False
        else:
            self.save_image_flag = True

        self.logger.debug(f'saving image data to {self.image_directory}')
        fname = f'{self.image_directory}/img_{time.time()}.h5'
        with h5py.File(fname, 'w') as f:
            for name, item in data_dict.items():
                if 'images' in name:
                    f.create_dataset(name, data=item)

                else:
                    if item is not None:
                        f['/'].attrs[name] = item

    def _get_images(self):
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
        min_size = 100
        rms_x = []
        rms_y = []
        ellipses = []
        n_blobs = []
        processed_images = []
        image_check = []
        centroid_offset = []
        for i in range(len(roi_images)):
            processed_image_data = image_processing.process_and_fit(roi_images[i],
                                                                    min_size)
            rms_x += [processed_image_data['rms_x']]
            rms_y += [processed_image_data['rms_y']]
            ellipses += [processed_image_data['ellipses']]
            n_blobs += [processed_image_data['n_blobs']]
            processed_images += [processed_image_data['smoothed_image']]
            centroid_offset += [processed_image_data['centroid_offset']]

            image_check += [image_processing.check_image(processed_image_data['binary_image'],
                                                         processed_image_data['smoothed_image'])]

        outputs = {'ellipses': np.array(ellipses), 'processed_images': np.array(processed_images),
                   'rms_x': np.array(rms_x), 'rms_y': np.array(rms_y),
                   'n_blobs': np.array(n_blobs), 'image_check': np.array(image_check),
                   'centroid_offset': np.array(centroid_offset),
                   }

        # add in raw data
        outputs.update(raw_outputs)

        return outputs

    def measure_screen(self):
        """
        Do screen measurement using controller_interface

        """
        outputs = self._get_images()

        if self.save_image_flag:
            self.save_images(outputs)
        return outputs
