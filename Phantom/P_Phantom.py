import numpy as np 

from BaseClass.Phantom_Base import *

'''create a 'P' shape matrix to simulate the MPI reconstruction'''

class Phantom_Shape_P(Phantom_Base):
    def __init__(self,temperature=20.0,diameter=30e-9,saturation_mag_core=8e5,concentration=5e7):
        super().__init__(diameter,saturation_mag_core,temperature,concentration)

    def get_Shape(self):  
        x, y = self._X, self._Y
        shape = np.zeros((y, x))

        shape[int(y * (14 / 120)):int(y * (105 / 120)), int(x * (29 / 120)):int(x * (90 / 120))] =\
        np.ones((int(y * (105 / 120)) - int(y * (14 / 120)),int(x * (90 / 120)) - int(x * (29 / 120)))) 
        shape[int(y * (29 / 120)):int(y * (60 / 120)), int(x * (44 / 120)):int(x * (75 / 120))] =\
        np.zeros((int(y * (60 / 120)) - int(y * (29 / 120)),int(x * (75 / 120)) - int(x * (44 / 120))))
        shape[int(y * (74 / 120)):int(y * (105 / 120)), int(x * (44 / 120)):int(x * (90 / 120))] =\
        np.zeros((int(y * (105 / 120)) - int(y * (74 / 120)),int(x * (90 / 120)) - int(x * (44 / 120))))

        return shape

    def get_Picture(self):
        # 可以矩阵Hadamard积形式计算，也可以广播形式计算
        return self._Shape * self._concentration_value
        

