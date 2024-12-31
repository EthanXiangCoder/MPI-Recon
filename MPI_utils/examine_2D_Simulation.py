from BaseClass.Constant_Base import *

import numpy as np


class Information_examine():
    def __init__(self,message):
        self.Data_Cut(message)
        judge, str1, str2 = self.Data_Check(message)
        if judge is False:
            print(f"message_{str1}_{str2} have some problems")

    def data_check(self,varible,type):
        '''单个数据'''
        if isinstance(varible,type):
            return True
        elif type==float and isinstance(varible, int):
            return True
        else:
            return False
    
    def data_check_2(self,varible,type_list):
        '''单个数据 有数据类型可供选择'''
        if str(varible.dtype) in type_list:
            return True
        else:
            return False
    
    def Data_Check(self,message):
        '''
            check the type 
        ''' 
        # Part1: Particle Porperty
        if not message['Particle_Porperty']['Diameter'] is None:
            flag = self.data_check(message['Particle_Porperty']['Diameter'], float)
            if flag == False:
                return False,'Particle_Porperty','Diameter'

        if not message['Particle_Porperty']['Temperature'] is None:
            flag = self.data_check(message['Particle_Porperty']['Temperature'], float)
            if flag == False:
                return False,'Particle_Porperty','Temperature'     

        if not message['Particle_Porperty']['Saturation_Mag'] is None:
            flag = self.data_check(message['Particle_Porperty']['Saturation_Mag'], float)
            if flag == False:
                return False,'Particle_Porperty','Saturation_Mag'    

        # Part2: Selection Field
        if not message['Selection_Field']['X_Gradient'] is None:
            flag = self.data_check(message['Selection_Field']['X_Gradient'],float)
            if flag == False:
                return False,'Selection_Field','X_Gradient' 

        if not message['Selection_Field']['Y_Gradient'] is None:
            flag = self.data_check(message['Selection_Field']['Y_Gradient'],float)
            if flag == False:
                return False,'Selection_Field','Y_Gradient'

        # if not message['Selection_Field']['Z_Gradient'] is None:
        #     flag = self.data_check(message['Selection_Field']['Z_Gradient'],float)
        #     if flag == False:
        #         return False,'Selection_Field','Z_Gradient'

        # Part3: Drive Field
        if not message['Drive_Field']['X_Waveform'] is None:
            flag1 = self.data_check_2(message['Drive_Field']['X_Waveform']['X_Amplitude'],["float","float32","float64","float128","int","int32","int64","int128"])
            flag2 = self.data_check_2(message['Drive_Field']['X_Waveform']['X_Frequency'],["float","float32","float64","float128","int","int32","int64","int128"])
            flag3 = self.data_check_2(message['Drive_Field']['X_Waveform']['X_Phase'],["float","float32","float64","float128","int","int32","int64","int128"])
            if not (flag1 and flag2 and flag3):
                return False,'Drive_Field','X_Waveform'
            
        if not message['Drive_Field']['Y_Waveform'] is None:
            flag1 = self.data_check_2(message['Drive_Field']['Y_Waveform']['Y_Amplitude'],["float","float32","float64","float128","int","int32","int64","int128"])
            flag2 = self.data_check_2(message['Drive_Field']['Y_Waveform']['Y_Frequency'],["float","float32","float64","float128","int","int32","int64","int128"])
            flag3 = self.data_check_2(message['Drive_Field']['Y_Waveform']['Y_Phase'],["float","float32","float64","float128","int","int32","int64","int128"])
            if not (flag1 and flag2 and flag3):
                return False,'Drive_Field','Y_Waveform'
        
        # if not message['Drive_Field']['Z_Waveform'] is None:
        #     flag1 = self.data_check_2(message['Drive_Field']['Z_Waveform']['Z_Amplitude'],["float","float32","float64","float128","int","int32","int64","int128"])
        #     flag2 = self.data_check_2(message['Drive_Field']['Z_Waveform']['Z_Frequency'],["float","float32","float64","float128","int","int32","int64","int128"])
        #     flag3 = self.data_check_2(message['Drive_Field']['Z_Waveform']['Z_Phase'],["float","float32","float64","float128","int","int32","int64","int128"])
        #     if not (flag1 and flag2 and flag3):
        #         return False,'Drive_Field','Z_Waveform'
        
        if not message['Drive_Field']['RepeatTime'] is None:
            flag = self.data_check(message['Drive_Field']['RepeatTime'],float)
            if flag == False:
                return False,'Drive_Field','RepeatTime'
        
        if not message['Drive_Field']['WaveType'] is None:
            flag = self.data_check(message['Drive_Field']['WaveType'],str)
            if flag == False:
                return False,'Drive_Field','WaveType'
        
        # Part4: Focus Field
        # if not message['Focus_Field']['X_Direction'] is None:
        #     flag = self.data_check_2(message['Focus_Field']['X_Direction'],["float","float32","float64","float128"])
        #     if flag == False:
        #         return False,'Focus_Field','X_Direction'
        
        # if not message['Focus_Field']['Y_Direction'] is None:
        #     flag = self.data_check_2(message['Focus_Field']['Y_Direction'],["float","float32","float64","float128"])
        #     if flag == False:
        #         return False,'Focus_Field','Y_Direction'
        
        # if not message['Focus_Field']['Z_Direction'] is None:
        #     flag = self.data_check_2(message['Focus_Field']['Z_Direction'],["float","float32","float64","float128"])
        #     if flag == False:
        #         return False,'Focus_Field','Z_Direction'
            
        # if not message['Focus_Field']['WaveType'] is None:
        #     flag = self.data_check(message['Focus_Field']['WaveType'],str)
        #     if flag == False:
        #         return False,'Focus_Field','WaveType'
            
        # Part5: Information about Sample
        if not message['Sample']['Topology'] is None:
            flag = self.data_check(message['Sample']['Topology'],str)
            if flag == False:
                return False,'Sample','Topology'

        if not message['Sample']['Sample_Trajectory'] is None:
            flag = self.data_check(message['Sample']['Sample_Trajectory'],str)
            if flag == False:
                return False,'Sample','Sample_Trajectory'

        if not message['Sample']['Frequency'] is None:
            flag = self.data_check_2(message['Sample']['Frequency'],["float","float32","float64","float128","int","int32","int64","int128"])
            if flag == False:
                return False,'Sample','Frequency'

        if not message['Sample']['Sample_Number'] is None:
            flag = self.data_check(message['Sample']['Sample_Number'],int)
            if flag == False:
                return False,'Sample','Sample_Number'

        if not message['Sample']['Sample_Time'] is None:
            flag = self.data_check_2(message['Sample']['Sample_Time'],["float","float32","float64","float128","int","int32","int64","int128"])
            if flag == False:
                return False,'Sample','Sample_Time'
        
        # Part6: Information about Measurement
        if not message['Measurement']['Recon_Type'] is None:
            flag = self.data_check(message['Measurement']['Recon_Type'],str)
            if flag == False:
                return False,'Measurement','Recon_Type'
        
        if not message['Measurement']['Measure_Signal'] is None:
            flag = self.data_check_2(message['Measurement']['Measure_Signal'][0][10],["float","float32","float64","float128","int","int32","int64","int128"]) #random choose
            if flag == False:
                return False,'Measurement','Measure_Signal'
            
        if not message['Measurement']['Auxiliary_Information'] is None:
            flag = self.data_check_2(message['Measurement']['Auxiliary_Information'][0][10],["float","float32","float64","float128","int","int32","int64","int128","complex64","complex128"]) #random choose
            if flag == False:
                return False,'Measurement','Auxiliary_Information'
        
        if not message['Measurement']['Voxel_Number'] is None:
            flag = self.data_check(message['Measurement']['Voxel_Number'][0],int)
            if flag == False:
                return False,'Measurement','Voxel_Number'
            
        if not message['Measurement']['Sensitivity'] is None:
            flag1 = self.data_check_2(message['Measurement']['Sensitivity']['X_Sensitivity'],["float","float32","float64","float128","int","int32","int64","int128"])
            flag2 = self.data_check_2(message['Measurement']['Sensitivity']['Y_Sensitivity'],["float","float32","float64","float128","int","int32","int64","int128"])
            #flag3 = self.data_check_2(message['Measurement']['Sensitivity']['Z_Sensitivity'],["float","float32","float64","float128","int","int32","int64","int128"])
            if not (flag1 and flag2):
                return False,'Measurement','Sensitivity'

        return True,'',''
    
    def Data_Normalization(self,message):
        '''
        this function is useless until we use it
        '''
        
        message['Measurement']['Measure_Signal'] /= KB

        if not message['Sample']['Sensetivity'] is None:
            message['Measurement']['Measure_Signal'] /= message['Sample']['Sensetivity']

        if (not message['Particle_Porperty']['Diameter'] is None) and (not message['Particle_Porperty']['Saturation_Mag'] is None):
            m = message['Particle_Porperty']['Saturation_Mag'] * (message['Particle_Porperty']['Diameter'] ** 3) * PI / 6.0
            message['Measurement']['Measure_Signal'] /= m

            if not message['Particle_Porperty']['Temperature'] is None:
                b = U0 * m / (message['Particle_Porperty']['Temperature'] * KB)
                message['Measurement']['Measure_Signal'] /= b

    def Data_Cut(self,message):
        '''
            this function is not prepared for the reconstrution of the 2D Simulation
        '''
        pass




