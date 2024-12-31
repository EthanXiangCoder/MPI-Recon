import numpy as np
from tqdm import tqdm

from MPI_utils.weight_matrix import *
from MPI_utils.time import *

def initialize_lambda(A,energy_mat):
    '''
    O(M * N)'''

    timer = Time()

    M = A.shape[0]
    N = A.shape[1]
    A_conjugate = np.conjugate(np.transpose(A))
    lambda_number = np.double(0)

    # first_b = np.copy(A)
    # lambda_number = np.double(0)
    # for i in range(M):
    #     first_b[i,:] *= np.sqrt(energy_mat[i][i])
    # for j in range(N):
    #     temp = first_b[:,j]@np.conjugate(first_b[:,j])
    #     lambda_number += temp

    for i in tqdm(range(N),desc="initialize_lambda computation"):
        for j in range(M):
            lambda_number += A_conjugate[i][j] * energy_mat[j][j] * A[j][i]

    timer.cal_time()
    print(f"initialize_lambda: {timer.time[-1]}s")
    timer.reset()

    return lambda_number / N