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
acc_n.shape

# Convert specific columns to values
acc_s = acc_s.to_numpy()
acc_s = np.transpose(acc_s)
acc_s.shape

# acc_n(:,1) = C*acc_s(:,1) # matlab

acc_n[:, 0] = np.matmul(C, acc_s[:, 0])
acc_n[:, 0]
acc_n.shape


# %%
