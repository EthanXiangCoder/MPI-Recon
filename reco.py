import numpy as np 
import h5py
import urllib
import os
import sys
import matplotlib.pyplot as plt
import urllib.request
from mpl_toolkits.axes_grid1 import make_axes_locatable

from test_python.kaczmarzReg import *
from test_python.pseudoinverse import *

from MPI_utils.weight_matrix import *
from MPI_utils.initailize_lambda import *
from MPI_utils.time import *
from MPI_utils.computation_with_timeout import *

SM_path = 'systemMatrix_V2.mdf'
Measurment_path = 'measurement_V2.mdf'

def download_progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading: {percent}%", end="")

def download_file(url, filename):
    try:
        urllib.request.urlretrieve(url, filename, reporthook=download_progress_hook)
        print(f"\nDownloaded {filename} successfully.")
    except Exception as e:
        print(f"\nFailed to download {filename}: {e}")

if not os.path.isfile(SM_path):
    print("Downloading systemMatrix_V2.mdf...")
    download_file('http://media.tuhh.de/ibi/mdfv2/systemMatrix_V2.mdf', SM_path)

if not os.path.isfile(Measurment_path):
    print("Downloading measurement_V2.mdf...")
    download_file('http://media.tuhh.de/ibi/mdfv2/measurement_V2.mdf', Measurment_path)

fSM = h5py.File(SM_path, 'r')
fMeas = h5py.File(Measurment_path, 'r')

SM = fSM['/measurement/data']
print(SM.shape) # (1*3*817*1959) J*C*K*N
SM = SM[:,:,:,:].squeeze()
isBG = fSM['/measurement/isBackgroundFrame'][:].view(bool)
SM = SM[:,:,isBG == False]
print(SM.shape) # (3*817*1936) isBG == 23

U = fMeas['/measurement/data']
print(U.shape) # (500*1*3*1632) N*J*C*W      W equals to V
U = U[:,:,:,:].squeeze()
U = np.fft.rfft(U, axis=2)
print(U.shape) # (500*3*817)

# select frequency range according to DFT
sampling_num = fMeas['/acquisition/receiver/numSamplingPoints'][()]
frequency_num = round(sampling_num / 2) + 1
bandwidth = fMeas['/acquisition/receiver/bandwidth'][()]
freq_range = np.arange(0, frequency_num) / frequency_num * bandwidth
min_freq = np.where(freq_range > 80e3)[0][0] # attention: the return value of np.where is a tuple
SM = SM[0:2, min_freq:, :]
U = U[:, 0:2, min_freq:]

assert SM.shape[1] == U.shape[2], "The frequency dimension of SM and U is not the same." #2*K*1959 500*2*K

SM = np.reshape(SM, (SM.shape[0] * SM.shape[1], SM.shape[2])) # 2K*1959
U = np.reshape(U, (U.shape[0], U.shape[1] * U.shape[2])) # 500*2K
U = np.mean(U, axis=0) # 1*2K

# # calculate the concentration
# beta = np.linalg.norm(SM, ord='fro') * 1e-3
# c_kac = kaczmarzReg(SM, U, iterations=10, lambd=beta, enforceReal=False, enforcePositive=True, shuffle=True)
# U_svd, S, V = np.linalg.svd(SM, full_matrices=False)
# print(U_svd.shape, S.shape, V.shape)
# c_svd = pseudoinverse(U_svd, S, V, U, 5e2, enforceReal=True, enforcePositive=True)
# print(c_kac.shape, c_svd.shape)

# # reshape the concentration into an image
# size = fSM['/calibration/size'][:]
# c_kac = np.reshape(c_kac, (size[0], size[1]))
# c_svd = np.reshape(c_svd, (size[0], size[1]))

# fig, ax = plt.subplots(1, 2,figsize=(10,5))
# # 绘制第一个图像
# im1 = ax[0].imshow(np.real(c_kac), cmap="plasma", vmin=-0.2, vmax=1.2)
# divider1 = make_axes_locatable(ax[0])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(im1, cax=cax1)
# # 绘制第二个图像
# im2 = ax[1].imshow(np.real(c_svd), cmap="plasma", vmin=-0.2, vmax=1.2)
# divider2 = make_axes_locatable(ax[1])
# cax2 = divider2.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(im2, cax=cax2)

# plt.show()

Matrix = energy_matrix(SM)
Matrix_2 = normalization_matrix(SM)
Matrix_3 = identity_matrix(SM)

Lambda = initialize_lambda(SM, Matrix)
print(Lambda)
Lambda_2 = initialize_lambda(SM, Matrix_2)
print(Lambda_2)
Lambda_3 = initialize_lambda(SM, Matrix_3)
print(Lambda_3)

Iterations = 30
Enforce_Rule = True

def _Kaczmarz_origin(A,b,Lambda):
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

    for i in tqdm(range(Iterations),desc="Reconstruction Calculation"):
        for j in range(M):
            k = rowIndexCycle[j]
            alpha = (b[k] - np.dot(A[k,:].conjugate(),x) - np.sqrt(Lambda) * v[k]) / (energy[k] ** 2 + Lambda)

            x += alpha * A[k,:]
            v[k] += alpha * np.sqrt(Lambda)

    timer.cal_time()
    print(f"reconstrution time: ", timer.time[-1])
    timer.reset() 

    x.imag = 0
    x = x * (x.real > 0)

    return x

def _Kaczmarz_withweight(A,b,Lambda,Matrix):
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

    for i in tqdm(range(Iterations),desc="Reconstruction Calculation"):
        for j in range(M):
            k = rowIndexCycle[j]
            alpha = (b[k] - np.dot(A[k,:].conjugate(),x) - np.sqrt(Lambda / Matrix[k][k]) * v[k]) / (energy[k] ** 2 + Lambda / Matrix[k][k])

            x += alpha * A[k,:]
            v[k] += alpha * np.sqrt(Lambda)
            if Enforce_Rule:
                if np.iscomplexobj(x):
                    x.imag = 0
                x = x * (x.real > 0)

    timer.cal_time()
    print(f"reconstrution time:", timer.time[-1])
    timer.reset()

    x.imag = 0
    x = x * (x.real > 0)
    return x

def _ImageReshape(c,size):
    y = size[0]
    x = size[1]
    c = np.flip(c)
    c = np.real(np.reshape(c,(y,x)))
    # c /= np.max(c) ####
    return c

c1 = _Kaczmarz_withweight(SM,U,0.001,Matrix)
c1 = _ImageReshape(c1, (44,44))

c2 = _Kaczmarz_origin(SM,U,0.001)
c2 = _ImageReshape(c2, (44,44))

fig, ax = plt.subplots(1, 2,figsize=(10,5))

im1 = ax[0].imshow(np.real(c1), cmap="plasma", vmin=-0.2, vmax=1.2)
divider1 = make_axes_locatable(ax[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)
im2 = ax[1].imshow(np.real(c2), cmap="plasma", vmin=-0.2, vmax=1.2)
divider2 = make_axes_locatable(ax[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

plt.show()


