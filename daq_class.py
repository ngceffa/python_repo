import numpy as np
# These are the libraries needed to work with NI DAQ module.
import nidaqmx
import nidaqmx.stream_writers
# ------------------------------------------------------------------------------
class analogOut(object):
    
    def __init__(self, device_name='Dev1', channel_name='ao0'):
    
        """ Constructor:
                - device_name = obvious;
                - channel_name = obvious;   
        """ 
        self.device_name = device_name
        self.channel_name = channel_name       
        self.task = nidaqmx.Task(self.channel_name)
        self.task.ao_channels.add_ao_voltage_chan(device_name+'/'+channel_name)
# ------------------------------------------------------------------------------
    def constant(self, value):
        assert(value<=10. or value >-10.), \
                                    '\n max output 10V: you put   %f' %value
        self.task.write(value)

# ------------------------------------------------------------------------------
    def sine(self, frequency = 1, V_max = 1, V_min = -1,\
                    num_samples = 10**4):
                    
        """ Generate sin wave:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - V_min = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <=10.), '\n max output 10V: you put   %f' %V_max
        assert(V_max >=-10.), '\n max output 10V: you put %f' %V_min
                    
        self.t = np.linspace(0, 1, int(num_samples/frequency))
        
        self.signal = (np.sin(np.pi*self.t/.5)+1) * (V_max-V_min+1)/2. + V_min
        
        self.task.timing.cfg_samp_clk_timing(num_samples,\
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan= num_samples)
        self.task.write(self.signal)
        self.task.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.task.start()
        
# ------------------------------------------------------------------------------
    def stop(self):
        """ It stops the movement.
        """
        self.task.stop()
        self.task.close()
        self.task = nidaqmx.Task(self.channel_name)
        self.task.ao_channels.add_ao_voltage_chan(\
                            self.device_name+'/'+self.channel_name)
# ------------------------------------------------------------------------------
    def close(self):
        """ It closes the task, i.e. the galvo object stops working at all.
        """
        self.task.stop()
        self.task.close()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------





















