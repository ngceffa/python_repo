import sys
import importlib
sys.path.append('C:\Program Files\Micro-Manager-2.0beta')
import MMCorePy
import numpy as np
import math_functions as mtm
importlib.reload(mtm)
import time
from threading import Thread
import skimage.external.tifffile as tiffile
import galvos_with_shutter as daq
import os
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class camera(Thread):
    """Camera initialization
    """
    def __init__(self, initialization):
        Thread.__init__(self)
        # Load all parameters to start camera communication
        self.mmc = MMCorePy.CMMCore()
        self.name = initialization[0]
        self.mmc.loadDevice(*initialization)
        self.mmc.initializeDevice(self.name)
        self.mmc.setCameraDevice(self.name)

        # vectorial notation: x = number of rows, y = num. of col
        self.x = self.mmc.getImageHeight()
        self.y = self.mmc.getImageWidth()

        self.stack = np.zeros((1, self.x, self.y))
        self.roi_width = int(self.mmc.getImageHeight())
        self.roi_height = int(self.mmc.getImageWidth())
        self.roi_x0 = 0
        self.roi_y0 = 0
        self.iter_roi_x0 = 0
        self.iter_roi_y0 = 0
        self.converted_flag = True
        self.snapped = np.zeros((self.x,self.y), dtype='uint32')
        self.num_points=5

        self.stack = []
        self.acquire_flag = True
        self.savedata_path = 'E:\\DATA\\Laura\\Ablation\\'
        print(self.savedata_path)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def print_properties(self):
        """ Prints all the camera properties and all the possible vaues of
        these properties
        """
        prop_ro = list()
        print ('\nDEVICE PROPERTIES LIST:\n')
        for prop in self.mmc.getDevicePropertyNames(self.name):
            if self.mmc.isPropertyReadOnly(self.name, prop):
                prop_ro.append(prop)
            else:
                low = self.mmc.getPropertyLowerLimit(self.name, prop)
                up =self.mmc.getPropertyUpperLimit(self.name, prop)
                available_vals=\
                ', '.join(self.mmc.getAllowedPropertyValues(self.name, prop))
                if(available_vals):
                    print(str(prop) +'= '+ \
                                    self.mmc.getProperty(self.name, prop)+\
                                    ' --> possibe values from this set: {' \
                                                    + available_vals + '}\n')
                else:
                    print(str(prop) +'= '+ \
                                    self.mmc.getProperty(self.name, prop)\
                                    + ', choose from '+ \
                                            str(low)+ ' to ' + str(up) +' \n')
        print ( '\nRead-only prop:\n'), prop_ro
        return None

    def snap(self, snap_label = 'default'):
        ''' Snap an image.
        '''
        self.mmc.clearCircularBuffer()
        time.sleep(.1)
        self. mmc.snapImage()
        time.sleep(.1)
        self.snapped = self.mmc.getImage()
        self.snapped=np.fliplr(self.snapped)
        return None

    def volumes_acquisition(self, delta_lines = 100,
                        line_speed = 10000,\
                        number_of_images = 300, freq = 1,
                        V_max = 1.3, V_min = -1.3,\
                        alpha = 84., z_V_min= -0.5, z_V_max = 0.5, \
                        modality = 'volumes'):

        self.mmc.setCircularBufferMemoryFootprint(1000)
        #SET UP
        # the first command is needed to directly control scan speed
        self.mmc.setProperty(self.name, 'TriggerMode', 'External')
        self.mmc.setProperty(self.name, 'TransposeMirrorX', 1)
        self.mmc.setProperty(self.name,\
                                'LightScanPlus-ScanSpeedControlEnable', 'On')
        self.mmc.setProperty(self.name,\
                                'LightScanPlus-SensorReadoutMode', \
                                'Bottom Up Sequential')
        self.mmc.setProperty(self.name,\
                                'LightScanPlus-LineScanSpeed [lines/sec]', \
                                str(line_speed))
        self.mmc.setProperty(self.name,\
                                'LightScanPlus-ExposedPixelHeight', \
                                str(delta_lines))

        freq = 1./((2048+delta_lines+5)/line_speed)
        print('acquisition frequency is: ' + str(freq))

        # ACQUISITION PROCEDURE
        self.acquire_flag = True
        self.mmc.startContinuousSequenceAcquisition(0)
        time.sleep(.5)
        self.scan = daq.multi_DAQ_analogOut()
        self.run_threads(modality)
        self.scan.acquisition_signals(freq)
        return None

    def saving(self):
        time.sleep(.5)
        while(self.acquire_flag == True):
            if(self.mmc.getBufferFreeCapacity()<
                                        self.mmc.getBufferTotalCapacity()):
                while(self.mmc.getBufferFreeCapacity()<
                                        self.mmc.getBufferTotalCapacity()):
                     ##LM flip lr
                    self.stack.append(np.fliplr(self.mmc.popNextImage()))
                time.sleep(.1)
        return None
#----------------------------------------------------------------------------------------------

    def run_threads(self, what):
        if (what == 'planes'):
            print('I\'m a saving thread')
            thread = Thread(target=self.saving)
            thread.start()
        if (what == 'volumes'):
            print('I\'m a saving thread')
            thread = Thread(target=self.saving_matrices_as_tiff)
            thread.start()
        return None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def close(self):
        """ Closes the communication with the camera
        """
        self.mmc.reset()
        return None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TEST

# zyla = camera(['zyla', 'AndorSDK3', 'Andor sCMOS Camera'])
