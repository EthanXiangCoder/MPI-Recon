from abc import ABC,abstractmethod
from BaseClass.Constant_Base import *

'''message initialization'''

class Information_Class(ABC):
    def __init__(self):
        '''message初始化'''
        self.message = {
            particle_porperty:{
                diameter:None,
                temperature:None,
                saturation_mag:None
            },
            selection_field:{
                x_gradient:None,
                y_gradient:None,
                z_gradient:None
            },
            drive_field:{
                x_waveform:{
                    x_amplitude:None,
                    x_frequency:None,
                    x_phase:None
                },
                y_waveform:{
                    y_amplitude:None,
                    y_frequency:None,
                    y_phase:None
                },
                z_waveform:{
                    z_amplitude:None,
                    z_frequency:None,
                    z_phase:None
                },
                repeat_time:None, #轨迹周期时间
                wave_type:None
            },
            focus_field:{
                x_direction:{
                    x_amplitude:None,
                    x_frequency:None,
                    x_phase:None
                },
                y_direction:{
                    y_amplitude:None,
                    y_frequency:None,
                    y_phase:None
                },
                z_direction:{
                    z_amplitude:None,
                    z_frequency:None,
                    z_phase:None
                },
                wave_type:None,
            },
            sample:{
                topology:None, #FFP or FFL
                sample_trajectory:None, #Lissajous Trajectory or Cartesian Trajectory
                frequency:None, 
                sample_number:None,
                sample_time:None,
            },
            measurement:{
                sensitivity:{
                    x_sensitivity:None,
                    y_sensitivity:None,
                    z_sensitivity:None
                }, #coil sensitivity
                recon_type:None,
                measure_signal:None,
                # auxiliary_information based on sytstem matrix method will be tough for the main memory'''
                auxiliary_information:None,
                voxel_number:None,
            }
        }

    def _get_item1(self,messagefirst,messagesecond,content):
        self.message[messagefirst][messagesecond] = content

    def _get_item2(self,messagefirst,messagesecond,messagethird,content):
        self.message[messagefirst][messagesecond][messagethird] = content

