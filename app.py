# %%
import pandas as pd
import itertools
import numpy as np
from scipy.integrate import cumtrapz
from numpy import sin, cos, pi
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('seaborn')

df = pd.read_csv('1.csv', encoding='unicode_escape')
gyro_bias = [0.0066, -0.0071, 0.0120]

# df.head()
# df.describe()
# df.info()


data_size = len(df)
data_size

# timestamp = df.loc[:, "time"]
# timestamp

timestamp = df["time"]
timestamp

acc_s = df.loc[:, 'acc_x':'acc_z']
acc_s

gyro_s = df.loc[:, 'gyr_x': 'gyr_z']  # Rates of turn in sensor frame.
gyro_s

g = 9.8  # Gravity.

# Initialise parameters.
# Orientation from accelerometers. Sensor is assumed to be stationary.

pitch = -asin(acc_s.iloc[1, 1]/g)
roll = atan(acc_s.iloc[2, 1]/acc_s.iloc[3, 1])
yaw = 0


C = np.array([[cos(pitch)*cos(yaw), (sin(roll)*sin(pitch)*cos(yaw))-(cos(roll)*sin(yaw)), (cos(roll)*sin(pitch)*cos(yaw))+(sin(roll)*sin(yaw))],
              [cos(pitch)*sin(yaw), (sin(roll)*sin(pitch)*sin(yaw))+(cos(roll)
                                                                     * cos(yaw)), (cos(roll)*sin(pitch)*sin(yaw))-(sin(roll)*cos(yaw))],
              [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]])

C_prev = C
C_prev

heading = np.empty((1, data_size))
heading[:] = np.nan
heading[0, 0] = yaw
heading


# Gyroscope bias, to be determined for each sensor.
# Defined above so we don't forget to change for each dataset. --
# Preallocate storage for accelerations in navigation frame.


acc_n = np.empty((3, data_size))
acc_n[:] = np.nan
acc_n
acc_n.shape

# Convert the first columns from df to np.values
acc_s = acc_s.to_numpy()
acc_s = np.transpose(acc_s)
acc_s
acc_s.shape

# acc_n(:,1) = C*acc_s(:,1) # matlab

acc_n[:, 0] = np.matmul(C, acc_s[:, 0])
acc_n[:, 0].shape
acc_n[:, 0]


# % Preallocate storage for velocity (in navigation frame).
# % Initial velocity assumed to be zero.

vel_n = np.empty((3, data_size))
vel_n[:] = np.nan
vel_n

# vel_n(:,1) = [0 0 0]'; matlab
vel_n[:, 0] = np.nan_to_num(0)
vel_n


# Preallocate storage for position (in navigation frame).
# Initial position arbitrarily set to the origin.
pos_n = np.empty((3, data_size))
pos_n[:] = np.nan
pos_n

# % pos_n(: , 1) = [0 0 0]' matlab
pos_n[:, 0] = np.nan_to_num(0)
pos_n[:, 0]
pos_n


# Preallocate storage for distance travelled used for altitude plots.
distance = np.empty((1, data_size-1))
distance[:] = np.nan
distance.shape
distance[0, 0] = np.nan_to_num(0)
distance[0, 0]
distance


# %%
