import numpy as np

from abc import ABC,abstractmethod
from BaseClass.Constant_Base import *


class Phantom_Base(ABC):
    def __init__(self,diameter,saturation_mag_core,temperature,concentration): 
        self._diameter = diameter
        self._temperature = temperature+T_BASE  #Kelvin Temperature
        self._particle_volume = self.__get_volume()
        self._saturation_mag_core = saturation_mag_core
        self._particle_magnetic_moment = self.__get_particle_magnetic()
        self._concentration_value = concentration     #concentration is a matrix
        self._beta_coeff = self.__get_beta_coeff()

        self._X = None
        self._Y = None

        self._Shape = None 
        self._Picture = None 
        
        
    #calculate the volume of particle
    def __get_volume(self):
        return PI*(self._diameter**3) / 6.0
    
    #calculate the magnetic moment of a single particle
    def __get_particle_magnetic(self):
        return self._particle_volume*self._saturation_mag_core

    #calculate the coeff of beta (Langevin Function)
    def __get_beta_coeff(self):
        return U0 * self._particle_magnetic_moment/(KB * self._temperature)
    
    #return the phantom matrix
    # def get_Phantom(self,x,y,concentration):
    #     self.get_Shape()
    #     return self._Picture
    
    @abstractmethod
    def get_Shape(self):
        pass

    @abstractmethod
    def get_Picture(self):
        pass    