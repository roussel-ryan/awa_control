import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import filters, transform


def zero_surrounded(array):
    return not (array[0, :].any() or
                array[-1, :].any() or
                array[:, 0].any() or
                array[:, -1].any())


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)

    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


def process_and_fit(image, min_size=100, verbose=False):
    """
    Process an image with a region of interest specified
    - smooths the image with a 3 px Gaussian filter
    - thresholds the image using a triangle threshold algorithm
    - identifies blobs based on blob size, if blob size < min_size the blob is removed

    NOTE: all measurements are in pixels
    """
    logger = logging.getLogger(__name__)

    smoothed_image = filters.gaussian(image, 3)

    triangle_threshold = filters.threshold_triangle(smoothed_image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')
    binary_image = np.where(smoothed_image > triangle_threshold, 1, 0)

    binary_image = remove_small_blobs(binary_image, min_size)
    binary_image = binary_image.astype(np.uint8)

    smoothed_image = np.where(binary_image,
                              smoothed_image, 0)


    # find contours
    cnts, huers = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # fit ellipses
    ellipses = []
    for cnt in cnts:
        blob_area = cv2.contourArea(cnt)
        logger.debug(f'blob area: {blob_area}')
        if blob_area > min_size:
            ellipse = cv2.fitEllipse(cnt)
            ellipses += [ellipse]

    n_blobs = len(ellipses)

    proj_x = np.sum(smoothed_image, axis=0)
    proj_y = np.sum(smoothed_image, axis=1)

    # calculate stds and means
    x_len = len(proj_x)
    y_len = len(proj_y)
    mean_x, rms_x = weighted_std(np.arange(x_len), proj_x)
    mean_y, rms_y = weighted_std(np.arange(y_len), proj_y)

    # calculate distance to center
    beam_center = np.array((mean_x, mean_y))
    image_center = np.array(smoothed_image.shape) / 2
    distance_to_center = np.linalg.norm(image_center - beam_center)

    if verbose:
        fig, ax = plt.subplots(1, 3)
        c = ax[0].imshow(image)
        ax[1].imshow(smoothed_image)
        ax[2].imshow(binary_image)

        fig.colorbar(c)
        plt.show()

    output = {'binary_image': binary_image,
              'smoothed_image': smoothed_image,
              'n_blobs': n_blobs,
              'ellipses': ellipses,
              'centroid_offset': distance_to_center,
              'mean_x': mean_x,
              'mean_y': mean_y,
              'rms_x': rms_x,
              'rms_y': rms_y}

    return output


def check_image(binary_image, simage):
    """
    Note: use this with a sub-image ROI

    check for the following
    - if the beam is entirely inside the ROI
    - if the image is saturated


    """
    logger = logging.getLogger(__name__)

    if np.count_nonzero(simage > 65400) > 200:
        logger.warning(f'image saturated')
        return 0

    # if there is no beam on the edges
    if zero_surrounded(binary_image):
        return 1
    else:
        logger.warning('ROI is clipping the beam envelope')
        return 0


def remove_small_blobs(image, min_size):
    labels, n_blobs = ndimage.label(image)
    new_image = image.copy()

    for i in range(1, n_blobs + 1):
        counts = np.count_nonzero(i == labels)
        if counts < min_size:
            new_image = np.where(i == labels, 0, new_image)

    return new_image


def rotate_beamlets(image):
    logger = logging.getLogger(__name__)
    timage, simage, n_blobs, ellipses, _, _ = process_and_fit(image, min_size=500)

    mean_angle = np.mean(np.array([ele[-1] for ele in ellipses]))
    rotated_image = transform.rotate(simage, mean_angle - 90.0)

    return rotated_image, mean_angle - 90.0, n_blobs


if __name__ == '__main__':
    data = np.genfromtxt('../AWA/test_images/onenc250200.txt', names=True)

    size = 50e-3
    bins = 700
    img, xedges, yedges = np.histogram2d(data['x'], data['y'],
                                         range=np.array(((-0.0, 1.0),
                                                         (-0.0, 1.0))) * size / 2.0,
                                         bins=(bins, bins))

    img = img / np.max(img)
    img = 0.15 * np.random.rand(*img.shape) + img

    print(check_image(img))
    plt.show()
