import numpy as np

def row_energy(A):
    M = A.shape[0]
    energy = np.zeros(M, dtype = np.double)

    for m in range(M):
        energy[m] = np.linalg.norm(A[m,:]) 
    return energy

def identity_matrix(A):
    M = A.shape[0]
    return np.eye(N = M,dtype = np.double)

def energy_matrix(A):
    energy= row_energy(A)
    M = A.shape[0]
    eye_origin = np.eye(N = M,dtype = np.double)
    for i in range(M):
        eye_origin[i,i] *= energy[i]
    return eye_origin 

def normalization_matrix(A):
    energy= row_energy(A)
    M = A.shape[0]
    eye_origin = np.eye(N = M,dtype = np.double)
    for i in range(M):
        eye_origin[i,i] /= (energy[i] ** 2)
    return eye_origin