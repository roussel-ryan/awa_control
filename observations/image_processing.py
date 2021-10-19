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


def process_and_fit(image, min_size=100, min_intensity=0.05, verbose=False):
    """
    Process an image with a region of interest specified
    - smooths the image with a 3 px Gaussian filter
    - thresholds the image using a triangle threshold algorithm
    - identifies blobs based on blob size, if blob size < min_size the blob is removed

    NOTE: all measurements are in pixels
    """
    logger = logging.getLogger(__name__)

    # normalize image
    image = image / np.max(image)
    
    # smooth image
    smoothed_image = filters.gaussian(image, 3)


    triangle_threshold = filters.threshold_triangle(smoothed_image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')
    binary_map = np.where(smoothed_image > triangle_threshold, 1, 0)
    image = np.where(binary_map, image, 0)
    
    # threshold based on blob size and intensity
    post_processed_image, intensities = remove_small_and_dim_blobs(image, binary_map, min_size, min_intensity)    
    labels, n_blobs = ndimage.label(post_processed_image)
    
    total_intensity = np.sum(post_processed_image)
    
    #------
    if verbose:
        fig, ax = plt.subplots(1, 4)
        c = ax[0].imshow(image)
        ax[1].imshow(post_processed_image)
        c = ax[2].imshow(post_processed_image > 0)
        ax[3].plot(np.sum(post_processed_image, axis=1),range(post_processed_image.shape[0])[::-1])

        fig.colorbar(c)
        plt.show()
    
    # find contours
    cnts, huers = cv2.findContours(post_processed_image.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # fit ellipses
    ellipses = []
    for cnt in cnts:
        blob_area = cv2.contourArea(cnt)
        logger.debug(f'blob area: {blob_area}')
        if blob_area > min_size:
            ellipse = cv2.fitEllipse(cnt)
            ellipses += [ellipse]


    proj_x = np.sum(post_processed_image, axis=0)
    proj_y = np.sum(post_processed_image, axis=1)

    # calculate stds and means
    x_len = len(proj_x)
    y_len = len(proj_y)
    mean_x, rms_x = weighted_std(np.arange(x_len), proj_x)
    mean_y, rms_y = weighted_std(np.arange(y_len), proj_y)

    # calculate distance to center
    beam_center = np.array((mean_x, mean_y))
    image_center = np.array(post_processed_image.shape) / 2
    distance_to_center = np.linalg.norm(image_center - beam_center)

    
    output = {'binary_image': post_processed_image > 0,
              'n_blobs': n_blobs,
              'ellipses': ellipses,
              'centroid_offset': distance_to_center,
              'mean_x': mean_x,
              'mean_y': mean_y,
              'rms_x': rms_x,
              'rms_y': rms_y,
              'total_intensity': total_intensity,
              'blob_intensities': intensities,
              'post_processed_image': post_processed_image}
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


def remove_small_and_dim_blobs(raw_image, binary_map, min_size, min_intensity):
    labels, n_blobs = ndimage.label(binary_map)
    new_image = raw_image.copy()
    intensities = np.empty(n_blobs)
    for i in range(1, n_blobs + 1):
        counts = np.count_nonzero(i == labels)
        intensity = np.sum(np.where(i == labels, new_image, 0))
        intensities[i-1] = intensity
        if counts < min_size or intensity < min_intensity: #or if sum(over pixels) 
            new_image = np.where(i == labels, 0, new_image)

    return new_image, intensities

def remove_dim_blobs(image, binary, min_int):
    labels, n_blobs = ndimage.label(image)
    new_binimage = binary.copy()
    logger = logging.getLogger(__name__)
    intensities=np.empty(n_blobs)
    for i in range(1, n_blobs + 1):
        intensities[i-1] = np.sum(image[labels==i])
        logger.debug('blob '+str(i)+' intensity = '+str(intensities[i-1]))
        if intensities[i-1] < min_int: 
            new_binimage = np.where(i == labels, 0, new_binimage)
            logger.debug('blob '+str(i)+' eliminated')

    return new_binimage, intensities


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

    #print(check_image(img))
    plt.show()
