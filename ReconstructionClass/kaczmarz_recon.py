import numpy as np
from tqdm import tqdm

from BaseClass.Constant_Base import *

from MPI_utils.weight_matrix import *
from MPI_utils.initailize_lambda import *
from MPI_utils.time import *
from MPI_utils.computation_with_timeout import *


class SM_recon_Kaczmarz(object):
    def __init__(self,Message,Weight_type,Recon_type=2,Iterations=10,Enforce_Rule=True,Lambda=1e-6): 
        '''
        Attributes:
            Recon_type: 1--no weight matrix, 2--with weight matrix
            Iterations: the number of iterations
            Enforce_Rule: whether enforce the concentration to be positive and no imaginary part
            Weight_type: 1--energy matrix, 2--identity matrix, 3--normalization matrix
            Lambda: the regularization parameter'''

        self._Image_data = []
        self._Iterations = Iterations
        self._Enforce_Rule = Enforce_Rule

        timer = Time() 

        if Weight_type == 1:
            self._Matrix = energy_matrix(Message[measurement][auxiliary_information])
        elif Weight_type == 2:
            self._Matrix = identity_matrix(Message[measurement][auxiliary_information])
        elif Weight_type == 3:
            self._Matrix = normalization_matrix(Message[measurement][auxiliary_information])
        else:
            raise Exception("Weight_type is wrong !!!")
        
        timer.cal_time()
        print(f"matrix_time", timer.time[-1])
        timer.reset()

        if Lambda:
            self._Lambda = Lambda
        else:
            self._Lambda = initialize_lambda(Message[measurement][auxiliary_information],self._Matrix)

        if Recon_type == 1:
            self._ImageRecon1(Message[measurement][auxiliary_information],Message[measurement][measure_signal],Message[measurement][voxel_number])
        else:
            self._ImageRecon2(Message[measurement][auxiliary_information],Message[measurement][measure_signal],Message[measurement][voxel_number])
        

    def _Kaczmarz_origin(self,A,b):
        '''
        random choose the row of system matrix 
        not enforce the concentration to be positive and no imaginary part
        no weight matrix and some tricks about denoising'''
        energy = row_energy(A)
        M = A.shape[0]
        N = A.shape[1]

        x = np.zeros(N, dtype = b.dtype)
        v = np.zeros(M, dtype = b.dtype)

        rowIndexCycle = np.arange(0,M)
        np.random.shuffle(rowIndexCycle)
        timer = Time()

        for i in tqdm(range(self._Iterations),desc="Reconstruction Calculation"):
            for j in range(M):
                k = rowIndexCycle[j]
                alpha = (b[0][k] - np.dot(A[k,:].conjugate(),x) - np.sqrt(self._Lambda) * v[k]) / (energy[k] ** 2 + self._Lambda)

                x += alpha * A[k,:]
                v[k] += alpha * np.sqrt(self._Lambda)
            
        timer.cal_time()
        print(f"reconstrution time: ", timer.time[-1])
        timer.reset()    
        
        x.imag = 0
        x = x * (x.real > 0)

        return x

    def _Kaczmarz_withweight(self,A,b):
        '''
        random choose the row of system matrix 
        can choose whether enforce the concentration to be positive and no imaginary part
        has weight matrix and some tricks about denoising'''
        energy = row_energy(A)
        M = A.shape[0]
        N = A.shape[1]

        x = np.zeros(N, dtype = b.dtype)
        v = np.zeros(M, dtype = b.dtype)

        rowIndexCycle = np.arange(0,M)
        np.random.shuffle(rowIndexCycle)
        timer = Time()

        for i in tqdm(range(self._Iterations),desc="Reconstruction Calculation"):
            for j in range(M):
                k = rowIndexCycle[j]
                alpha = (b[0][k] - np.dot(A[k,:].conjugate(),x) - np.sqrt(self._Lambda / self._Matrix[k][k]) * v[k]) / (energy[k] ** 2 + self._Lambda / self._Matrix[k][k])

                x += alpha * A[k,:]
                v[k] += alpha * np.sqrt(self._Lambda)
                if self._Enforce_Rule:
                    if np.iscomplexobj(x):
                        x.imag = 0
                    x = x * (x.real > 0)
            
        timer.cal_time()
        print(f"reconstrution time:", timer.time[-1])
        timer.reset()
        
        x.imag = 0
        x = x * (x.real > 0)
        return x


    def _ImageReshape(self,c,size):
        y = size[0]
        x = size[1]
        c = np.flip(c)
        c = np.real(np.reshape(c,(y,x)))
        c /= np.max(c) ####
        return c
    
    def _ImageRecon1(self,A,b,size):
        self._Image_data.append(self._Kaczmarz_origin(A,b))
        self._Image_data.append(self._ImageReshape(self._Image_data[0],size))
    
    def _ImageRecon2(self,A,b,size):
        self._Image_data.append(self._Kaczmarz_withweight(A,b))
        self._Image_data.append(self._ImageReshape(self._Image_data[0],size))

    def get_Image(self):
        return self._Image_data    
