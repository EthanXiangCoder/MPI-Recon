import numpy as np

'''
    prepare for the constant list for the rest reconstruction code
'''

# dataloading and saving file

Result_Path = "result_path"
Simulation_Path = "simulation_path"
Experiment_Path = "experiment_path"

V2_SM_Path = "systemMatrix_V2.mdf"
V2_Mea_Path = "measurement_V2.mdf"




# constant list needed in the code

PI = np.pi
KB = 1.3806488e-23
T_BASE = 273.15
U0 = 4.0 * PI *1e-7
EPS = 1e-11


# attribute in the information base

particle_porperty = 'Particle_Porperty'
diameter = 'Diameter'
temperature = 'Temperature'
saturation_mag = 'Saturation_Mag'

selection_field = 'Selection_Field'
x_gradient = 'X_Gradient'
y_gradient = 'Y_Gradient'
z_gradient = 'Z_Gradient'

drive_field = 'Drive_Field'
x_waveform = 'X_Waveform'
x_amplitude = 'X_Amplitude'
x_frequency = 'X_Frequency'
x_phase = 'X_Phase'
y_waveform = 'Y_Waveform'
y_amplitude = 'Y_Amplitude'
y_frequency = 'Y_Frequency'
y_phase = 'Y_Phase'
z_waveform = 'Z_Waveform'
z_amplitude = 'Z_Amplitude'
z_frequency = 'Z_Frequency'
z_phase = 'Z_Phase'
repeat_time = 'RepeatTime'
wave_type = 'WaveType'

focus_field = 'Focus_Field'
x_direction = 'X_Direction'
x_amplitude = 'X_Amplitude'
x_frequency = 'X_Frequency'
x_phase = 'X_Phase'
y_direction = 'Y_Direction'
y_amplitude = 'Y_Amplitude'
y_frequency = 'Y_Frequency'
y_phase = 'Y_Phase'
z_direction = 'Z_Direction'
z_amplitude = 'Z_Amplitude'
z_frequency = 'Z_Frequency'
z_phase = 'Z_Phase'
wave_type = 'WaveType'

sample = 'Sample'
topology = 'Topology'
sample_trajectory = 'Sample_Trajectory'
frequency = 'Frequency'
sample_number = 'Sample_Number'
sample_time = 'Sample_Time'

measurement = 'Measurement'
sensitivity = 'Sensitivity'
x_sensitivity = 'X_Sensitivity'
y_sensitivity = 'Y_Sensitivity'
z_sensitivity = 'Z_Sensitivity'
recon_type = 'Recon_Type'
measure_signal = 'Measure_Signal'
auxiliary_information = 'Auxiliary_Information'
voxel_number = 'Voxel_Number'
