import matplotlib.pyplot as plt
import numpy as np
import logging

from scipy.signal import find_peaks
from skimage import filters


logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def calculate_emittance(image, px_scale, slit_sep_m, R12):
    """
    calculate emittance using processed beam image

    Arguments
    ---------
    image : ndarray, size (N,M)
        Post-processed screen image with screen removed and noise present

    px_scale : float
        Pixel scaling factor in meters per pixel

    slit_sep_m : float
        Slit seperation in meters

    R12 : float
        R12 matrix element, usually the drift length in meters

    """


    # get projection
    orig_proj = np.sum(image, axis=1)
    orig_proj = orig_proj / np.max(orig_proj)

    # apply threshold
    # proj = np.where(proj > threshold * np.max(proj), proj, 0.0)
    triangle_threshold = None
    try:
        triangle_threshold = filters.threshold_triangle(orig_proj)
    except ValueError:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(orig_proj)
        ax[1].imshow(image)
        #print(image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')
    proj = np.where(orig_proj > 0, orig_proj, 0)


    # we assume that the beam is divergent, as a result the peaks should be at least
    # 2 mm apart
    peaks, _ = find_peaks(proj, distance=0.5e-3 / px_scale)

    if len(peaks) < 5:
        logger.warning(f'detected only {len(peaks)} peaks '
                       '-- emittance might be underestimated')

    # plot proj if in debugging
    if 0:
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        ax[0].imshow(image)
        ax[1].plot(orig_proj)
        ax[2].plot(proj)
        for pk in peaks:
            ax[2].axvline(pk)

    logger.debug(f'peak finding found {len(peaks)} peaks')
    logger.debug(f'found peaks at {peaks} px')
    logger.debug(f'mean seperation {np.mean(peaks[1:] - peaks[:-1]) * px_scale:.2e}')
    logger.debug(f'rms seperation {np.std(peaks[1:] - peaks[:-1]) * px_scale:.2e}')

    # calculate mid points and number of blobs
    mid_pts = (peaks[1:] + peaks[:-1]) / 2
    mid_pts = mid_pts.astype(np.int)
    mid_pts = np.array([0, *mid_pts, len(proj)])

    x_screen_px = np.arange(len(proj))
    n_blobs = len(mid_pts) - 1

    # calculate gaussian fit stats
    a = np.empty(n_blobs)  # relative intensity
    b = np.empty(n_blobs)  # 'y' peak position
    c = np.empty(n_blobs)  # 'y' sigma

    for n in range(n_blobs):
        sub_proj = proj[mid_pts[n]: mid_pts[n + 1]]
        sub_x = x_screen_px[mid_pts[n]: mid_pts[n + 1]]

        a[n] = np.sum(sub_proj)
        b[n] = np.average(sub_x, weights=sub_proj)
        c[n] = np.sqrt(np.average((sub_x - b[n]) ** 2, weights=sub_proj))

    # sort peaks by central position
    sorted_idx = np.argsort(b)
    a, b, c = a[sorted_idx], b[sorted_idx], c[sorted_idx]

    logger.debug(f'mean beamlet width_px: {np.mean(c)}')

    # convert pixel lengths to meters
    b = b * px_scale
    c = c * px_scale

    logger.debug(f'a: {a}')
    logger.debug(f'b: {b}')
    logger.debug(f'c: {c}')

    # define slit locations
    x_slit_m = np.linspace(-(n_blobs - 1) / 2 * slit_sep_m, (n_blobs - 1) / 2 * slit_sep_m, n_blobs)
    logger.debug(f'slit locations (m): {x_slit_m}')
    # calculate beam centroid at slits
    ixi = np.sum(a * x_slit_m) / np.sum(a)

    # center b coords on beam center on screen
    b = b - np.average(b, weights=a)
    logger.debug(b)

    # calculate beam size at slits - weighted rms of beamlet positions
    ixxi = np.sum(a * (x_slit_m - ixi) ** 2) / np.sum(a)

    # calc mean divergence (xp) of each beamlet
    xp = (b - x_slit_m) / R12

    # calc mean divergence of beam at slits -> should be approx 0 for perfectly aligned beam
    ixpi = np.sum(a * xp) / np.sum(a)

    # calc rms divergence of each beamlet (sp) at the screen c = R_12 * sp
    sp = c / R12

    # calc rms divergence at slits - weighted average of the divergence of each beamlet (sp)
    ixpxpi = np.sum(a * sp ** 2) / np.sum(a)
    # ixpxpi = np.sum(a * sp**2 + a * (xp - ixpi)**2) / np.sum(a)

    # calc correlation term at slits
    ixxpi = (np.sum(a * x_slit_m * xp) - np.sum(a) * ixi * ixpi) / np.sum(a)

    logger.debug('emittance calculation results')
    logger.debug(f'ixi:{ixi}')
    logger.debug(f'ixpi:{ixpi}')
    logger.debug(f'ixxi:{ixxi}')
    logger.debug(f'ixpxpi:{ixpxpi}')
    logger.debug(f'ixxpi:{ixxpi}')

    # calculate emittance
    # emittance = np.sqrt(ixxi * ixpxpi - ixxpi**2)
    emittance = np.sqrt(ixxi * ixpxpi)
    logger.info(f'calculated emittance: {emittance:.2e}, n_peaks:{len(peaks)}')

    return emittance


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # testing
    minscreen2mm = np.genfromtxt('test_images/onenc400200.txt', names=True)
    img, _, _ = np.histogram2d(minscreen2mm['x'], minscreen2mm['y'],
                               range=np.array(((-25.0, 25.0),
                                               (-25.0, 25.0))) * 1e-3,
                               bins=(700, 700))

    scale = 50e-3 / 700
    drift = 1.5748
    slit_sep = 0.002

    calculate_emittance(img.T, scale, slit_sep, drift)
