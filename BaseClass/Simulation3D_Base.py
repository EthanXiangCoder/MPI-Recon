import numpy as np
from matplotlib import pyplot as plt 
import os
import tqdm
from abc import ABC,abstractmethod

from BaseClass.Constant_Base import *
from BaseClass.Information_Base import *

class SimulationBase(Information_Class,ABC):

    '''
    SimulationBase: The base class of the Simulation component.'''
    def __init__(self,
                Phantom,
                SelectionField_X,
                SelectionField_Y,
                SelectionField_Z,
                DriveField_XA,
                DriveField_YA,
                DriveField_ZA,
                DriveField_XF,
                DriveField_YF,
                DriveField_ZF,
                DriveField_XP, #Phase PI/2.0之类的格式
                DriveField_YP,
                DriveField_ZP,
                Repeat_Time,
                Sample_Frequency,
                Move_StepSize):
        #考虑把trajectory作为参数传入 不再需要设置多组DriveField参数 将其作为一个比例参数传入
        super().__init__()

        self._phantom = Phantom #phantom class instance

        #coil sensitivity
        self._X_Sensiticity = 1.0 
        self._Y_Sensitivity = 1.0
        self._Z_Sensitivity = 1.0
        self._Sensiticity = np.array([self._X_Sensiticity,self._Y_Sensitivity,self._Z_Sensitivity]).reshape(-1,1)

        #selection field property
        self._selection_x = SelectionField_X / U0 
        self._selection_y = SelectionField_Y / U0
        self._selection_z = SelectionField_Z / U0
        self._selection_gradient = np.array([self._selection_x,self._selection_y,self._selection_z]).reshape(-1,1) 

        #drive field property
        self._drive_x_a = DriveField_XA / U0
        self._drive_y_a = DriveField_YA / U0
        self._drive_z_a = DriveField_ZA / U0
        self._drive_x_f = DriveField_XF
        self._drive_y_f = DriveField_YF
        self._drive_z_f = DriveField_ZF
        self._drive_x_p = DriveField_XP
        self._drive_y_p = DriveField_YP
        self._drive_z_p = DriveField_ZP
        self._drive_repeat_time = Repeat_Time 

        #FOV Def
        self._max_X = self._drive_x_a / self._selection_x
        self._max_Y = self._drive_y_a / self._selection_y
        self._max_Z = self._drive_z_a / self._selection_z

        #采样时间频率、采样时间列表、采样数量
        self._sample_frequency = Sample_Frequency
        self._sample_num = int(self._drive_repeat_time * self._sample_frequency) #前闭后开
        self._Tsequence = np.arange(0, self._drive_repeat_time, 1 / self._sample_frequency)

        #坐标位置初始化
        self.move_step_size = Move_StepSize
        self._Xsequence = np.arange(-self._max_X,self._max_X+self.move_step_size,self.move_step_size)
        self._Ysequence = np.arange(-self._max_Y,self._max_Y+self.move_step_size,self.move_step_size)
        self._Zsequence = np.arange(-self._max_Z,self._max_Z+self.move_step_size,self.move_step_size)
        self._X_num = len(self._Xsequence)
        self._Y_num = len(self._Ysequence)
        self._Z_num = len(self._Zsequence)

        self._phantom._X = self._X_num
        self._phantom._Y = self._Y_num
        self._phantom._Z = self._Z_num

        self._phantom._Shape = self._phantom.get_Shape()
        self._phantom._Picture = self._phantom.get_Picture()

        #计算所有时刻的驱动场场强
        self._X_Drive_sequence,self._X_Drive_Derivative_sequence = self.__Calculate_DriveField_Value(self._drive_x_a,self._drive_x_f,self._drive_x_p,self._Tsequence)
        self._Y_Drive_sequence,self._Y_Drive_Derivative_sequence = self.__Calculate_DriveField_Value(self._drive_y_a,self._drive_y_f,self._drive_y_p,self._Tsequence)
        self._Z_Drive_sequence,self._Z_Drive_Derivative_sequence = self.__Calculate_DriveField_Value(self._drive_z_a,self._drive_z_f,self._drive_z_p,self._Tsequence)
        self._Drive_sequence = np.array([self._X_Drive_sequence,self._Y_Drive_sequence,self._Z_Drive_sequence]) # 3*n
        self._Drive_Derivative_sequence = np.array([self._X_Drive_Derivative_sequence,self._Y_Drive_Derivative_sequence,self._Z_Drive_Derivative_sequence]) # 3*n

        #计算所有时刻的FFP位置
        self._Location = np.divide(self._Drive_sequence,np.tile((-1.0) * self._selection_gradient,(1,len(self._Tsequence)))) 
        self._Ordinate = self._Location / self.move_step_size
        self._X_Ordinate = self._Ordinate[0]  
        self._Y_Ordinate = self._Ordinate[1]
        self._Z_Ordinate = self._Ordinate[2]

        '''计算机坐标转换'''
        self.PC_Ordinate = self.__Mapping()
        
        #计算各个位置的梯度场强度
        self._selection_mag = self.__init_Gradient_mag()

    #计算驱动场在指定时间的场强 
    def __Calculate_DriveField_Value(self,Amplitude,Frequency,phase,Tsequence):
        Drive_sequence = Amplitude * np.cos(2.0 * PI * Frequency * Tsequence + phase) * (-1.0) 
        Drive_Derivative_sequence = Amplitude * 2.0 * PI * Frequency * np.sin(2.0 * PI * Frequency * Tsequence + phase) 
        return Drive_sequence,Drive_Derivative_sequence

    #坐标转换
    def __Mapping(self):
        '''
        得根据3D FOV的磁场梯度来计算 有两种情况'''
        pc_ordinate = np.zeros((3,len(self._Tsequence)))
        pc_ordinate[0] = self._max_X + self._X_Ordinate
        pc_ordinate[1] = (-1) * self._Y_Ordinate
        pc_ordinate[1] += self._max_Y
        return pc_ordinate
    
    # 根据画出的图形进行坐标系转换计算每个位置的梯度场强度
    def __init_Gradient_mag(self):
        Seletion_Mag = np.zeros((self._Y_num,self._X_num,2))
        for i in range(self._Y_num):
            y = (-1.0) * i * self.move_step_size + self._max_Y
            for j in range(self._X_num):
                x = j * self.move_step_size - self._max_X
                location = np.array([x,y]).reshape(-1,1)
                magnetic = self._selection_gradient * location
                Seletion_Mag[i,j,0] = magnetic[0]  
                Seletion_Mag[i,j,1] = magnetic[1]

        return Seletion_Mag 

    def Langevin_derivative_2D(self,m):
        # Langevin_derivative_matrix = np.zeros_like(m)
        # y, x = m.shape[0], m.shape[1]
        # for i in range(y):
        #     for j in range(x):
        #         for k in range(2):
        #             if(abs(m[i,j,k])<1):
        #                 Langevin_derivative_matrix[i,j,k] = 1.0 / 3.0
        #             else:
        #                 Langevin_derivative_matrix[i,j,k] = (1.0 / ((self._phantom._beta_coeff * m[i,j,k]) ** 2)) - \
        #                                                     (1.0 / ((np.sinh(self._phantom._beta_coeff * m[i,j,k])) ** 2))

        Langevin_derivative_matrix = (1.0 / np.power(self._phantom._beta_coeff * m,2)) - (1.0 / np.power(np.sinh(self._phantom._beta_coeff * m),2))

        return Langevin_derivative_matrix
    
    def _get_Voltage(self):

        voltage = np.zeros((2, self._sample_num))

        #self._Total_Field_Strength_length = np.zeros((self._Y_num, self._X_num, self._sample_num))
        self._Total_Field_Vector = np.zeros((self._Y_num, self._X_num, 2, self._sample_num))

        for i in range(self._sample_num):
            '''基于Langevin粒子模型和互易定律计算电压值'''
            coeff = (-1.0) * U0 * self._phantom._particle_magnetic_moment * self._phantom._beta_coeff * self._Drive_Derivative_sequence[:,i].reshape(-1,1) * self._Sensiticity

            drive_field_strength = np.tile(self._Drive_sequence[:,i],(self._Y_num,self._X_num,1))
            total_field_strength = np.add(drive_field_strength, self._selection_mag)

            #存储各个时刻的磁场强度值方便后续计算
            self._Total_Field_Vector[:,:,:,i] = total_field_strength

            '''
                这部分对于磁化特性响应导数曲线的部分，因为点乘运算直接将其转换为两部分，不再计算磁场强度模长
            '''
            # total_mod = np.sqrt(np.add(total_field_strength[:,:,0] ** 2,total_field_strength[:,:,1] ** 2))
            # Langevin_derivative = np.zeros((self._Y_num,self._X_num,2))
            # Langevin_derivative = (1.0 /  ((self._phantom._beta_coeff * total_mod) ** 2)) - \
            #                       (1.0 / ((np.sinh(self._phantom._beta_coeff * total_mod)) ** 2))

            Langevin_derivative = self.Langevin_derivative_2D(total_field_strength)
                
            phantom_shape = np.tile(self._phantom._Picture[:,:,np.newaxis],(1,1,2))

            intergal_result = phantom_shape * Langevin_derivative

            voltage[0,i] = np.sum(intergal_result[:,:,0]) * coeff[0] * self.move_step_size * self.move_step_size
            voltage[1,i] = np.sum(intergal_result[:,:,1]) * coeff[1] * self.move_step_size * self.move_step_size

        # os.makedirs(Result_Path,exist_ok=True)
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
        '''根据aux_type给出的方法确定是否需要转换电压信号和重建方法'''
        voltage = self._get_Voltage()
        aux_signal = self._get_AuxSignal()

        return voltage,aux_signal
    
    #初始化仿真相关数据
    def _send_2_information(self,trajectory,aux_type):
        voltage,aux_signal = self._get_Signal()

        self._get_item1('Particle_Porperty','Diameter',self._phantom._diameter)
        self._get_item1('Particle_Porperty','Temperature',self._phantom._temperature)
        self._get_item1('Particle_Porperty','Saturation_Mag',self._phantom._saturation_mag_core)

        self._get_item1('Selection_Field','X_Gradient',self._selection_x)
        self._get_item1('Selection_Field','Y_Gradient',self._selection_y)
        
        self._get_item2('Drive_Field','X_Waveform','X_Amplitude',self._drive_x_a)
        self._get_item2('Drive_Field','X_Waveform','X_Frequency',self._drive_x_f)
        self._get_item2('Drive_Field','X_Waveform','X_Phase',self._drive_x_p)
        self._get_item2('Drive_Field','Y_Waveform','Y_Amplitude',self._drive_y_a)
        self._get_item2('Drive_Field','Y_Waveform','Y_Frequency',self._drive_y_f)
        self._get_item2('Drive_Field','Y_Waveform','Y_Phase',self._drive_y_p)
        self._get_item1('Drive_Field','RepeatTime',self._drive_repeat_time)
        self._get_item1('Drive_Field','WaveType','sine') # or triangle / dirac
        
        '''no focus field'''

        self._get_item1('Sample','Topology','FFP')
        self._get_item1('Sample','Frequency',self._sample_frequency)
        self._get_item1('Sample','Sample_Number',self._sample_num)
        self._get_item1('Sample','Sample_Time',self._drive_repeat_time)
        self._get_item1('Sample','Sample_Trajectory',trajectory)

        self._get_item1('Measurement','Recon_Type',aux_type)
        self._get_item2('Measurement','Sensitivity','X_Sensitivity',self._X_Sensiticity)
        self._get_item2('Measurement','Sensitivity','Y_Sensitivity',self._Y_Sensitivity)
        self._get_item1('Measurement','Measure_Signal',voltage)
        self._get_item1('Measurement','Auxiliary_Information',aux_signal)
        self._get_item1('Measurement','Voxel_Number',np.array([self._Y_num,self._X_num]))
        































