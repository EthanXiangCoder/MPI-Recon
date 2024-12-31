import numpy as np
from matplotlib import pyplot as plt 
import os
from tqdm import tqdm
from abc import ABC,abstractmethod

from BaseClass.Constant_Base import *
from BaseClass.Information_Base import *

class SM2D_SimulationBase(Information_Class,ABC):
    def __init__(self,
                Phantom,
                SelectionField_X,
                SelectionField_Y,
                DriveField_XA,
                DriveField_YA,
                DriveField_XF,
                DriveField_YF,
                DriveField_XP, #Phase PI/2.0之类的格式
                DriveField_YP, 
                Repeat_Time,
                Sample_Frequency,
                Move_StepSize,
                Recal,
                isInverse,
                ):
        super().__init__()

        self._phantom = Phantom #phantom class instance
        self._recal = Recal
        self._isInverse = isInverse
        if not self._isInverse:
            self._title_voltage = "Calculating the voltage in the Forward Problem"
        else:
            self._title_voltage = "Calculating the voltage in the Inverse Problem"

        #coil sensitivity
        self._X_Sensiticity = 1.0 
        self._Y_Sensitivity = 1.0
        self._Sensiticity = np.array([self._X_Sensiticity,self._Y_Sensitivity]).reshape(-1,1)

        #selection field property
        self._selection_x = SelectionField_X / U0 
        self._selection_y = SelectionField_Y / U0
        self._selection_gradient = np.array([self._selection_x,self._selection_y]).reshape(-1,1) 

        #drive field property
        self._drive_x_a = DriveField_XA / U0
        self._drive_y_a = DriveField_YA / U0
        self._drive_x_f = DriveField_XF
        self._drive_y_f = DriveField_YF
        self._drive_x_p = DriveField_XP
        self._drive_y_p = DriveField_YP
        self._repeat_time = Repeat_Time 

        #FOV 仿真中和Phantom对齐
        self._max_X = self._drive_x_a / self._selection_x
        self._max_Y = self._drive_y_a / self._selection_y

        #采样时间频率、采样时间列表、采样数量
        self._sample_frequency = Sample_Frequency
        time_interval = 1.0 / self._sample_frequency
        self._sample_num = int(self._repeat_time * self._sample_frequency) #前闭后开
        # self._Tsequence = np.arange(0, self._repeat_time - 1 / self._sample_frequency, 1 / self._sample_frequency) float error
        self._Tsequence = np.arange(0, self._sample_num,1.0) * time_interval

        #坐标位置初始化
        self.move_step_size = Move_StepSize
        self._Xsequence = np.arange(-self._max_X + self.move_step_size / 2,self._max_X,self.move_step_size)
        self._Ysequence = np.arange(-self._max_Y + self.move_step_size / 2,self._max_Y,self.move_step_size)
        self._X_num = len(self._Xsequence)
        self._Y_num = len(self._Ysequence)
        self._reverse_Ysequence = np.flip(self._Ysequence) #数组坐标系和

        #phantom初始化
        self._phantom._X = self._X_num
        self._phantom._Y = self._Y_num
        self._phantom._Shape = self._phantom.get_Shape()
        self._phantom._Picture = self._phantom.get_Picture()

        #计算所有时刻的驱动场场强
        self._Tsequence = self._Tsequence.reshape(-1,1) # (M*1)
        self._X_Drive_sequence,self._X_Drive_Derivative_sequence = self.__Calculate_DriveField_Value(self._drive_x_a,self._drive_x_f,self._drive_x_p,self._Tsequence)
        self._Y_Drive_sequence,self._Y_Drive_Derivative_sequence = self.__Calculate_DriveField_Value(self._drive_y_a,self._drive_y_f,self._drive_y_p,self._Tsequence)
        #self._Drive_sequence = np.array([self._X_Drive_sequence,self._Y_Drive_sequence])
        #self._Drive_Derivative_sequence = np.array([self._X_Drive_Derivative_sequence,self._Y_Drive_Derivative_sequence]) ]
        self._Drive_sequence = np.concatenate((self._X_Drive_sequence,self._Y_Drive_sequence),axis = 1) # (M*2)
        self._Drive_Derivative_sequence = np.concatenate((self._X_Drive_Derivative_sequence,self._Y_Drive_Derivative_sequence),axis = 1) # (M*2)


        #计算所有时刻的FFP位置和过扫描区域
        self._FFPLocation = self.__Calculate_FFPLocation( )
        self._OverScanRegion = np.zeros((self._Y_num * 2,self._X_num * 2)) # 


        #计算各个位置磁场强度
        self._MagLocation = self.__SetLocation()
        self._selection_mag = self.__init_Gradient_mag()
        #self._Total_Field_Vector = np.zeros((self._Y_num, self._X_num, 2, self._sample_num))

    #计算驱动场在指定时间的场强 
    def __Calculate_DriveField_Value(self,Amplitude,Frequency,phase,Tsequence):
        Drive_sequence = Amplitude * np.cos(2.0 * PI * Frequency * Tsequence + phase) * (-1.0) 
        Drive_Derivative_sequence = Amplitude * 2.0 * PI * Frequency * np.sin(2.0 * PI * Frequency * Tsequence + phase) 
        return Drive_sequence,Drive_Derivative_sequence

    #计算所有的FFP坐标位置 这部分可以和MDF中的过扫描区域对应
    def __Calculate_FFPLocation(self):
        Broad_selection_gradient = np.tile(self._selection_gradient,(1,self._sample_num)) # (M*2) 
        return np.divide(self._Drive_sequence,Broad_selection_gradient.T) # (M*2)
    
    def __isOverScanRegion(self):
        pass

    #坐标转换赋值
    def __SetLocation(self):
        mag_ordinate = np.zeros((self._Y_num,self._X_num,2))
        for i in range(self._Y_num):
           for j in range(self._X_num):
                mag_ordinate[i,j,0] = self._Xsequence[j]
                mag_ordinate[i,j,1] = self._reverse_Ysequence[i]
        return mag_ordinate 
    
    # 根据画出的图形进行坐标系转换计算每个位置的梯度场强度
    def __init_Gradient_mag(self):
        Seletion_Mag = np.zeros((self._Y_num,self._X_num,2))
        Seletion_Mag[:,:,0] = self._MagLocation[:,:,0] * self._selection_x
        Seletion_Mag[:,:,1] = self._MagLocation[:,:,1] * self._selection_y

        return Seletion_Mag 

    def Langevin_derivative_2D(self,m):
        return (1.0 / np.power(self._phantom._beta_coeff * m,2)) - (1.0 / np.power(np.sinh(self._phantom._beta_coeff * m),2))
    
    def _get_Voltage(self):
        '''
        only running within the SM reconstruction method!
        the particle response model is based on the Langevin Function'''

        voltage = np.zeros((2, self._sample_num))

        for i in tqdm(range(self._sample_num), desc=self._title_voltage):
            coeff = (-1.0) * U0 * self._phantom._particle_magnetic_moment * self._phantom._beta_coeff * self._Drive_Derivative_sequence[:,i].reshape(-1,1) * self._Sensiticity

            drive_field_strength = np.tile(self._Drive_sequence[:,i],(self._Y_num,self._X_num,1))
            total_field_strength = np.add(drive_field_strength, self._selection_mag)

            #存储各个时刻的磁场强度值方便后续计算
            self._Total_Field_Vector[:,:,:,i] = total_field_strength

            total_field_mod = np.sqrt(np.add(total_field_strength[:,:,0] ** 2,total_field_strength[:,:,1] ** 2))

            Langevin_derivative = self.Langevin_derivative_2D(total_field_mod)
            #phantom_shape = np.tile(self._phantom._Picture[:,:,np.newaxis],(1,1,2))

            intergal_result = self._phantom._Picture * Langevin_derivative

            voltage[0,i] = np.sum(intergal_result) * coeff[0] * self.move_step_size * self.move_step_size
            voltage[1,i] = np.sum(intergal_result) * coeff[1] * self.move_step_size * self.move_step_size

        # path = os.path.join(Result_Path,"voltage_time.png")
        # fig,axes = plt.subplots(nrows = 1,ncols = 2)
        # axes[0].plot(np.arange(0,self._sample_num),voltage[0,:])
        # axes[0].set_title("x-direction")
        # axes[1].plot(np.arange(0,self._sample_num),voltage[1,:])
        # axes[1].set_title("y-direction")
        # fig.savefig(path)

        return voltage
    
    @abstractmethod
    def _get_AuxSignal(self):
        pass

    @abstractmethod 
    def _translate_voltage(self,voltage):
        pass

    def _get_Signal(self):
        voltage = self._get_Voltage()
        voltage_transform = self._translate_voltage(voltage)
        aux_signal = self._get_AuxSignal()

        return voltage_transform,aux_signal
    
    #初始化仿真相关数据
    def _send_2_information(self,topology_type,trajectory,wave_type_input,aux_type,voltage=None,aux_signal=None):
        '''
        Attributes:
        topology_type: string  select the FFP or the FFL
        trajectory: string  select the trajectory of the FFP, such as Lissajous Trajectory
        wave_type_input: string  select the waveform of the drive field, such as sine wave
        aux_type: string  select the method of the reconstruction, such as reconstruction base on the system matrix'''
        if self._recal:
            voltage,aux_signal = self._get_Signal()

        self._get_item1(particle_porperty,diameter,self._phantom._diameter)
        self._get_item1(particle_porperty,temperature,self._phantom._temperature)
        self._get_item1(particle_porperty,saturation_mag,self._phantom._saturation_mag_core)

        self._get_item1(selection_field,x_gradient,self._selection_x)
        self._get_item1(selection_field,y_gradient,self._selection_y)
        
        self._get_item2(drive_field,x_waveform,x_amplitude,self._drive_x_a)
        self._get_item2(drive_field,x_waveform,x_frequency,self._drive_x_f)
        self._get_item2(drive_field,x_waveform,x_phase,self._drive_x_p)
        self._get_item2(drive_field,y_waveform,y_amplitude,self._drive_y_a)
        self._get_item2(drive_field,y_waveform,y_frequency,self._drive_y_f)
        self._get_item2(drive_field,y_waveform,y_phase,self._drive_y_p)
        self._get_item1(drive_field,repeat_time,self._repeat_time)
        self._get_item1(drive_field,wave_type,wave_type_input) # or triangle / dirac
        
        '''no focus field'''

        self._get_item1(sample,topology,topology_type)
        self._get_item1(sample,frequency,self._sample_frequency)
        self._get_item1(sample,sample_number,self._sample_num)
        self._get_item1(sample,sample_time,self._repeat_time)
        self._get_item1(sample,sample_trajectory,trajectory)

        self._get_item1(measurement,recon_type,aux_type)
        self._get_item2(measurement,sensitivity,x_sensitivity,self._X_Sensiticity)
        self._get_item2(measurement,sensitivity,y_sensitivity,self._Y_Sensitivity)
        self._get_item1(measurement,measure_signal,voltage)
        self._get_item1(measurement,auxiliary_information,aux_signal)
        self._get_item1(measurement,voxel_number,np.array([self._Y_num,self._X_num]))
        































