import matplotlib.pyplot as plt
from Phantom.P_Phantom import *

from BaseClass.Constant_Base import *
from SimulationClass.SM2D_simulation_Liss_FFP import *
from MPI_utils.examine_2D_Simulation import *   
from ReconstructionClass.kaczmarz_recon import *

import os

def Simulation_1(PhantomType, #phantom type(default 'P' shape)
                 relative_voltage_path,relative_auxsignal_path,
                 Recal,  #recalibration, this depends whether to calculate the simulation inverse
                 Temperature=20, Diameter=30e-9, Satur_Mag=8e5,Concentration=5e7,
                 Selection_X=2.0, Selection_Y=2.0,Drive_F_X=2500000.0/102.0,Drive_F_Y=2500000.0/96.0,Drive_A_X=12e-3,Drive_A_Y=12e-3,Drive_P_X=PI/2.0,Drive_P_Y=PI/2.0,
                 Sample_Time=6.528e-4,Sample_Frequency=2.5e6,
                 Delta_volume=1e-9,move_step_size=1e-4,inverse_step_size=1.2e-4,noise_rate=0.001,
                 inverse_concentration=None
                 ):
    ''' 2D Simulation based on System Matrix and Kaczmarz method'''
    
    try:
        if PhantomType == 0:
            phantom = Phantom_Shape_P(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=Concentration)
            if inverse_concentration is not None:
                phantom_inverse = Phantom_Shape_P(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=inverse_concentration)
        else:
            raise Exception("There is no such phantom type !!!")
    except:
        Exception("Phantom Module is broken !!!")
    
    try:
        simulation = SM_simulation_2D_Liss_FFP_Measurement(Phantom=phantom, 
                                                           Recal=Recal,
                                                           isInverse=False,
                                                           SelectionField_X=Selection_X, SelectionField_Y=Selection_Y,
                                                           DriveField_XA=Drive_A_X, DriveField_YA=Drive_A_Y, DriveField_XF=Drive_F_X, DriveField_YF=Drive_F_Y, DriveField_XP=Drive_P_X, DriveField_YP=Drive_P_Y,
                                                           Repeat_Time=Sample_Time, Sample_Frequency=Sample_Frequency,
                                                           concentration_delta_volume=Delta_volume,
                                                           Move_StepSize=move_step_size,
                                                           noise_rate=noise_rate,
                                                           Relative_Voltage_Path=relative_voltage_path, Relative_AuxSignal_Path=relative_auxsignal_path) 
    except:
        Exception("Simulation Module is broken !!!")
    if inverse_concentration is None:
        simulation_inverse = SM_simulation_2D_Liss_FFP_Measurement(Phantom=phantom, 
                                                                Recal=Recal,
                                                                isInverse=True,
                                                                SelectionField_X=Selection_X, SelectionField_Y=Selection_Y,
                                                                DriveField_XA=Drive_A_X, DriveField_YA=Drive_A_Y, DriveField_XF=Drive_F_X, DriveField_YF=Drive_F_Y, DriveField_XP=Drive_P_X, DriveField_YP=Drive_P_Y,
                                                                Repeat_Time=Sample_Time, Sample_Frequency=Sample_Frequency,
                                                                concentration_delta_volume=Delta_volume,
                                                                Move_StepSize=inverse_step_size,
                                                                noise_rate=noise_rate,
                                                                Relative_Voltage_Path=relative_voltage_path, Relative_AuxSignal_Path=relative_auxsignal_path)
    else:
        simulation_inverse = SM_simulation_2D_Liss_FFP_Measurement(Phantom=phantom_inverse, 
                                                                Recal=Recal,
                                                                isInverse=True,
                                                                SelectionField_X=Selection_X, SelectionField_Y=Selection_Y,
                                                                DriveField_XA=Drive_A_X, DriveField_YA=Drive_A_Y, DriveField_XF=Drive_F_X, DriveField_YF=Drive_F_Y, DriveField_XP=Drive_P_X, DriveField_YP=Drive_P_Y,
                                                                Repeat_Time=Sample_Time, Sample_Frequency=Sample_Frequency,
                                                                concentration_delta_volume=Delta_volume,
                                                                Move_StepSize=inverse_step_size,
                                                                noise_rate=noise_rate,
                                                                Relative_Voltage_Path=relative_voltage_path, Relative_AuxSignal_Path=relative_auxsignal_path) 

    '''until now the information of voltage and auxsignal is stored in the Result_Path'''
    if Recal:
        voltage_correct = np.load(os.path.join(Result_Path,relative_voltage_path))
        simulation_inverse._get_item1(measurement,measure_signal,voltage_correct)

    ImgData1 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type= 2,Recon_type=1,Enforce_Rule=False) #identity matrix, no Enforce_Rule and original Kaczmarz method
    result1 = ImgData1.get_Image()[1]

    ImgData2 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type= 1, Recon_type=2,Enforce_Rule=False) #energy matrix, no Enforce_Rule and Regularization Kaczmarz method 
    result2 = ImgData2.get_Image()[1]       

    ImgData3 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type= 3, Recon_type=2,Enforce_Rule=False) #normalization matrix,no Enforce_Rule and Rugularization Kaczmarz method
    result3 = ImgData3.get_Image()[1]

    ImgData4 = SM_recon_Kaczmarz(simulation_inverse.message,Lambda=None,Weight_type= 2, Recon_type=1) #learnable lambda,identity matrix, no Enforce_Rule and original Kaczmarz method
    result4 = ImgData4.get_Image()[1]

    ImgData5 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type= 2, Recon_type=2,Enforce_Rule=True) #identity matrix, Enforce_Rule and original Kaczmarz method
    result5 = ImgData5.get_Image()[1]

    ImgData7 = SM_recon_Kaczmarz(simulation_inverse.message,Lambda=None,Weight_type= 1, Recon_type=2,Enforce_Rule=True) #learnable lambda, energy matrix, Enforce_Rule and Rugularization Kaczmarz method
    result7 = ImgData7.get_Image()[1]

    ImgData8 = SM_recon_Kaczmarz(simulation_inverse.message,Lambda=None,Weight_type= 3, Recon_type=2,Enforce_Rule=True) #learnable lambda, normalization matrix, Enforce_Rule and Rugularization Kaczmarz method
    result8 = ImgData8.get_Image()[1]

    return phantom.get_Picture() / Concentration, result1, result2, result3, result4, result5, result7, result8


if __name__ == "__main__":
    print("*" * 32)
    print("1: 2D MPI Simulation based on System Matrix and Kaczmarz method (default 'P' shape)")
    print("2: xxxxxxx")
    print("3: xxxxxxx")
    print("4: xxxxxxx")
    print("5: xxxxxxx")
    print("6: xxxxxxx")
    print("7: xxxxxxx")
    print("8: xxxxxxx")
    print("Q: Quit")
    print("*" * 32)

    tutor  = input("select your own methods: ")
    default = ["1","Q"]

    # The particle concentration is variable. During the simulation, the value can be assigned as a matrix.
    # The default reconstruction is performed on a 121*121 matrix. Therefore, a variable concentration is provided as a reference.  
    # Users can choose to plot different imaging results based on this.
    change_matrix = np.ones((121,121))
    change_matrix[0:30,:] = 0.5
    change_matrix[31:60,:] = 1
    change_matrix[61:90,:] = 2
    change_matrix[91:120,:] = 4
    change_matrix *= 5e7

    inverse_matrix = np.ones((101,101))
    inverse_matrix[0:25,:] = 0.5
    inverse_matrix[26:50,:] = 1
    inverse_matrix[51:75,:] = 2
    inverse_matrix[76:100,:] = 4
    inverse_matrix *= 5e7

    while True:
        try:
            judge = tutor in default
            m = 1 / int(judge)
        except ZeroDivisionError:
            raise Exception("no such option !!!")        
        if tutor == "Q":
            break
        if tutor == "1":
            phantom,image1,image2,image3,image4,image5,image6,image7 = Simulation_1(
                PhantomType=0,#Concentration=change_matrix,
                relative_voltage_path="P_shape/60dB_voltage.npy",relative_auxsignal_path="P_shape/60dB_auxsignal.npy",
                Recal=False,
                noise_rate=0.001,
            )
            """always need these parameters below:
            -----PhantomType, distribution, Concentration, relative_voltage_path, relative_auxsignal_path, Recal"""

            fig,axes = plt.subplots(nrows = 2,ncols = 4,figsize = (12,5))
            axes = axes.flatten()
            c1 = axes[0].imshow(phantom, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c1, ax = axes[0])
            axes[0].set_title("phantom",fontsize = 10)
            axes[0].axes.get_xaxis().set_visible(False)
            axes[0].axes.get_yaxis().set_visible(False)

            c2 = axes[1].imshow(image1, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c2, ax = axes[1])
            axes[1].set_title("Iden/NE/GL Recon",fontsize = 10)
            axes[1].axes.get_xaxis().set_visible(False)
            axes[1].axes.get_yaxis().set_visible(False)

            c3 = axes[2].imshow(image2, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c3, ax = axes[2])
            axes[2].set_title("Ener/NE/GL Recon",fontsize = 10)
            axes[2].axes.get_xaxis().set_visible(False)
            axes[2].axes.get_yaxis().set_visible(False)

            c4 = axes[3].imshow(image3, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c4, ax = axes[3])
            axes[3].set_title("Norm/NE/GL Recon",fontsize = 10)
            axes[3].axes.get_xaxis().set_visible(False)
            axes[3].axes.get_yaxis().set_visible(False)

            c5 = axes[4].imshow(image4, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c5, ax = axes[4])
            axes[4].set_title("Iden/NE/LL Recon",fontsize = 10)
            axes[4].axes.get_xaxis().set_visible(False)
            axes[4].axes.get_yaxis().set_visible(False)

            c6 = axes[5].imshow(image5, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c6, ax = axes[5])
            axes[5].set_title("Iden/E/GL Recon",fontsize = 10)
            axes[5].axes.get_xaxis().set_visible(False)
            axes[5].axes.get_yaxis().set_visible(False)

            c7 = axes[6].imshow(image6, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c7, ax = axes[6])
            axes[6].set_title("Ener/E/LL Recon",fontsize = 10)
            axes[6].axes.get_xaxis().set_visible(False)
            axes[6].axes.get_yaxis().set_visible(False)

            c8 = axes[7].imshow(image7, cmap = "plasma", vmin = -0.2, vmax = 1.2)
            fig.colorbar(c8, ax = axes[7])
            axes[7].set_title("Norm/E/LL Recon",fontsize = 10)
            axes[7].axes.get_xaxis().set_visible(False)
            axes[7].axes.get_yaxis().set_visible(False)

            fig_path = os.path.join("./result_path","P_shape/60dB_recon.png")
            fig.savefig(fig_path)

            plt.show()
            break

# co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
# fig.colorbar(co, ax=axs[0, 1])
# axs[0, 1].set_title('contourf()')


