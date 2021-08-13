import matplotlib.pyplot as plt
import numpy as np
import time

import socket
import pythoncom
from win32com import client
import select

from epics import caget, caput, caget_many
import logging


class ROIError(Exception):
    pass


class AWAInterface:
    """
    controller_interface class to connect to AWA control system

    """
    UseNIFG = False
    Initialized = False
    m_AWACameraHost = "127.0.0.1"
    m_AWAPGCamPort = 2019
    m_AWANIFGPort = 2029
    NewMeasurement = False
    FWHMX = 10000
    FMHMY = 10000
    FMHML = 10000
    TempVal = 0.0

    def __init__(self, use_frame_grabber=True, testing=False):

        super().__init__()
        self.testing = testing
        self.logger = logging.getLogger(__name__)

        if not testing:
            self.USBDIO = client.Dispatch('USBDIOCtrl.Application')
            self.ni_frame_grabber = client.Dispatch('NIFGCtrl')
            self.AWAPGCamera = client.Dispatch('AWAPGCamera.application')
            self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            self.logger.info('Starting controller_interface')
            self.initialize_connections(use_frame_grabber)
            self.logger.info('Done')

    def initialize_connections(self, use_frame_grabber):
        # image client

        if self.testing:
            self.Initialized = False
        else:
            if self.Initialized:
                self.logger.debug('Deleteing old objects')
                del self.ni_frame_grabber
                del self.AWAPGCamera
                self.m_CameraClient.close()
                self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            else:
                pass

            pythoncom.CoInitialize()
            self.UseNIFG = use_frame_grabber
            if not self.UseNIFG:
                self.logger.debug('Connecting to AWAPGCamera application')
                self.m_CameraPort = self.m_AWAPGCamPort
            else:
                self.logger.debug('Connecting to NI Frame Grabber application')
                self.m_CameraPort = self.m_AWANIFGPort
            self.m_CameraClient.connect(("127.0.0.1", self.m_CameraPort))
            self.m_CameraClient.setblocking(0)

            self.Initialized = True
        # setpoint client
        # self.AWABeamlineClient = socket.socket(socket.AF_INET, socket. SOCK_STREAM)

    def close(self):
        if not self.Initialized:
            return

        if self.UseNIFG:
            del self.ni_frame_grabber

        else:
            del self.AWAPGCamera

    def get_raw_image(self):
        if self.Initialized:
            if self.UseNIFG:
                return np.array(self.ni_frame_grabber.GetImage())
            else:
                return np.array(self.AWAPGCamera.GetImage)
        else:
            raise RuntimeError('Trying to retrieve an image before controller_interface is initialized!')

    def get_roi(self):
        if self.Initialized:
            if self.UseNIFG:
                module = self.ni_frame_grabber
            else:
                module = self.AWAPGCamera

            x1 = module.ROIX1
            x2 = module.ROIX2
            y1 = module.ROIY1
            y2 = module.ROIY2

            if np.any(np.array((np.abs(x2 - x1), np.abs(y2 - y1))) < 20):
                raise ROIError('ROI is not large enough!')

            return np.array(((x1, y1), (x2, y2)))
        else:
            raise RuntimeError('Trying to retrieve an image '
                               'before controller_interface is initialized!')

    def get_data(self, target_charge=-1, charge_deviation=0.1, n_samples=1):
        """
        get new image and charge data

        Arguments
        ---------
        target_charge : float, optional
            Target charge for valid observation in nC, if negative ignore.
            Default: -1 (ignore)

        charge_deviation : float, optional
            Fractional deviation from target charge on ICT1 allowed for valid
            observation. Default: 0.1

        n_samples : int
            Number of samples to take

        note - calculations of centroids and FWHM etc. are based on a region of
        interest, which might be changed by the user!

        Connect to camera broadcasting TCP port for notifications
        If it is timed out, then just download and return whatever
        image available
        In order to avoid the complication of TCP buffer cleanup
        we will simply close the connection and reopen it
        elf.
        """

        self.logger.debug(f'taking n samples {n_samples}')

        n_shots = 0
        img = []
        charge = np.empty((n_samples, 4))
        centroid_y = np.empty((n_samples, 1))
        centroid_x = np.empty((n_samples, 1))
        fwhml = np.empty((n_samples, 1))
        fwhmy = np.empty((n_samples, 1))
        fwhmx = np.empty((n_samples, 1))

        roi = None

        if not self.testing:
            self.logger.debug('restarting camera client')
            self.m_CameraClient.close()
            self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.m_CameraClient.connect(("127.0.0.1", self.m_CameraPort))
            self.m_CameraClient.setblocking(False)

        while n_shots < n_samples:
            if not self.testing:
                ready = select.select([self.m_CameraClient], [], [], 2)

                if ready[0]:
                    # gate measurement
                    self.USBDIO.SetReadyState(2, 1)

                    # check charge on ICT1 is within bounds or charge bounds is not
                    # specified (target charge < 0)
                    ict1_charge = np.abs(caget(f'AWAICTMon:Ch1'))
                    if (np.abs(ict1_charge - target_charge) <
                        np.abs(charge_deviation * target_charge)) or \
                            (target_charge < 0):

                        a = self.m_CameraClient.recv(1024)
                        # print(a)
                        b = "".join(chr(x) for x in a)
                        try:
                            c = eval(b)

                            fwhmx[n_shots] = c['FWHMX']
                            fwhmy[n_shots] = c['FWHMY']
                            fwhml[n_shots] = c['FWHML']
                            centroid_x[n_shots] = c['CX']
                            centroid_y[n_shots] = c['CY']
                            self.NewMeasurement = True
                            img += [self.get_raw_image()]

                            # get charge
                            for i in range(1, 5):
                                charge[n_shots, i - 1] = caget(
                                    f'AWAICTMon:Ch{i}')

                            # get ROI
                            roi = self.get_roi()

                            self.logger.debug(roi)

                            n_shots += 1

                        except SyntaxError:
                            self.logger.warning('sleeping!')
                            time.sleep(0.1)

                    else:
                        # if we are considering charge limits then print a warning
                        if target_charge > 0:
                            self.logger.warning(f'ICT1 charge:{ict1_charge} nC'
                                                f' is outside target range')
                            time.sleep(0.1)

                    self.USBDIO.SetReadyState(2, 0)

                else:
                    self.logger.warning('camera client not ready for data')

            else:
                # generate testing data
                img_data, data = self.get_test_data()
                img += [img_data]
                fwhmx[n_shots] = data
                fwhmx[n_shots] = data
                fwhml[n_shots] = data
                centroid_y[n_shots] = data
                centroid_x[n_shots] = data

                roi = np.array(((10, 10), (990, 990)))
                charge[n_shots] = np.ones(4)
                n_shots += 1

        img = np.array(img)

        output = {'FWHMX': fwhmx,
                  'FWHMY': fwhmy,
                  'FWHML': fwhml,
                  'CX': centroid_x,
                  'CY': centroid_y,
                  'charge': charge,
                  'raw_images': img,
                  'ROI': roi}

        return output

    def set_parameters(self, param_dict):
        if self.testing:
            return

        for name, val in param_dict.items():
            self.logger.debug('sending epics command')
            self.logger.debug(f'caput {name} {val}')
            status = caput(name, val)
            self.logger.debug(f'caput return status {status}')

    def get_parameters(self, parameters):
        if self.testing:
            return np.random.rand(len(parameters))

        vals = caget_many(parameters)
        return dict(zip(parameters, vals))

    @staticmethod
    def get_test_data():
        # data is particle coordinates
        data = np.load('test_data/test_screen.npy')

        size = 25e-3
        bins = 1000
        img, xedges, yedges = np.histogram2d(data['x'], data['y'],
                                             range=np.array(((-1.0, 1.0),
                                                             (-1.0, 1.0))) * size,
                                             bins=(bins, bins))

        img = img.T

        # add noise
        img = img / np.max(img)
        img = 0.0 * np.random.rand(*img.shape) + img

        return img, 1.0
