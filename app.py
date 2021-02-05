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

# Read data from csv file.
df = pd.read_csv('1.csv', encoding='unicode_escape')
gyro_bias = [-0.0156, -0.0101, -0.0020]
gyro_bias


df.head()
# df.describe()
# df.info()


data_size = len(df)
data_size

# timestamp = df.loc[:, "time"]
timestamp = df["time"]
timestamp

acc_s = df.loc[:, 'acc_x':'acc_z']
acc_s

gyro_s = df.loc[:, 'gyr_x': 'gyr_z']  # Rates of turn in sensor frame.
gyro_s

g = 9.8  # Gravity.

# Initialise parameters.
# Orientation from accelerometers. Sensor is assumed to be stationary.

pitch = -asin(acc_s.iloc[0, 0]/g)
pitch
roll = atan(acc_s.iloc[0, 1]/acc_s.iloc[0, 2])
roll
yaw = 0

C = np.array([[cos(pitch)*cos(yaw), (sin(roll)*sin(pitch)*cos(yaw))-(cos(roll)*sin(yaw)), (cos(roll)*sin(pitch)*cos(yaw))+(sin(roll)*sin(yaw))],
              [cos(pitch)*sin(yaw), (sin(roll)*sin(pitch)*sin(yaw))+(cos(roll) *
                                                                     cos(yaw)), (cos(roll)*sin(pitch)*sin(yaw))-(sin(roll)*cos(yaw))],
              [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]])

C_prev = C

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
acc_n[:, 0]
acc_n.shape


# Preallocate storage for velocity (in navigation frame).
# Initial velocity assumed to be zero.

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

# Error covariance matrix.
P = np.zeros((9, 9))
P.shape
P

# Process noise parameter, gyroscope and accelerometer noise.
sigma_omega = 1e-2
sigma_a = 1e-2
sigma_omega
sigma_a


# ZUPT measurement matrix.
H = np.eye(3, 9, k=6)
H

# ZUPT measurement noise covariance matrix.
sigma_v = 1e-2
sigma_v

# R = diag([sigma_v sigma_v sigma_v]).^2, matlab
R = np.zeros((3, 3))
np.fill_diagonal(R, sigma_v)
R = R**2
R

# Gyroscope stance phase detection threshold.
gyro_threshold = 0.6
gyro_threshold


# Main Loop

data_size = 2
for t in range(1, data_size):
    # Start INS (transformation, double integration)
    dt = timestamp[t] - timestamp[t-1]

    # Remove bias from gyro measurements.
    gyro_s1 = (gyro_s[t:t+1] - gyro_bias).to_numpy()

    # Skew-symmetric matrix for angular rates
    ang_rate_matrix = np.array([[0, -gyro_s1[0, 2], gyro_s1[0, 1]],
                                [gyro_s1[0, 2], 0, -gyro_s1[0, 0]],
                                [-gyro_s1[0, 1], gyro_s1[0, 0], 0]])

    # orientation esimation ** chrck point **
    C = C_prev*(2*np.eye(3)+(ang_rate_matrix*dt)) / \
        (2*np.eye(3)-(ang_rate_matrix*dt))

#     ------C_prev----------
#    0.873375795078730  -0.100056444275897   0.476658607946463
#                    0   0.978670780982292   0.205434910498463
#   -0.487046938775510  -0.179421878293523   0.854747371460731

    # -------(2*eye(3)--------
    #  2     0     0
    #  0     2     0
    #  0     0     2

# -------ang_rate_matrix*dt--------
#    1.0e-03 *

#                    0  -0.016893018000000   0.164021454000000
#    0.016893018000000                   0   0.111250566000000
#   -0.164021454000000  -0.111250566000000                   0

# ------a----------
# a = (2*eye(3))+(ang_rate_matrix*dt);
#    2.000000000000000  -0.000016893018000   0.000164021454000
#    0.000016893018000   2.000000000000000   0.000111250566000
#   -0.000164021454000  -0.000111250566000   2.000000000000000

# ------b----------
# b = (2*eye(3))-(ang_rate_matrix*dt);
#    2.000000000000000   0.000016893018000  -0.000164021454000
#   -0.000016893018000   2.000000000000000  -0.000111250566000
#    0.000164021454000   0.000111250566000   2.000000000000000

# ------c----------
# c = (((2*eye(3))+(ang_rate_matrix*dt))/((2*eye(3))-(ang_rate_matrix*dt)));
#    0.999999986405794  -0.000016902141573   0.000164020512699
#    0.000016883894093   0.999999993668969   0.000111251950308
#   -0.000164022392057  -0.000111249179491   0.999999980360137

# ------d----------
#  d = C_prev*((2*eye(3))+(ang_rate_matrix*dt));
#    1.746671717664210  -0.200180671044745   0.953449336924665
#   -0.000017163029603   1.957318707214514   0.410978698675238
#   -0.974237105434710  -0.358930620023205   1.709394895988884

    # Transforming the acceleration from sensor frame to navigation frame.
    acc_n[:, t] = 0.5*(C + C_prev)@acc_s[:, t]
    acc_n[:, t]

    # Velocity and position estimation using trapeze integration.
    vel_n[:, t] = vel_n[:, t-1] + \
        ((acc_n[:, t] - [0, 0, g])+(acc_n[:, t-1] - [0, 0, g]))*dt/2
    (vel_n[:, t])

    pos_n[:, t] = pos_n[:, t-1] + (vel_n[:, t] + vel_n[:, t-1])*dt/2

    # Skew-symmetric cross-product operator matrix formed from the n-frame accelerations.
    S = np.array([[0, -acc_n[2, t], acc_n[1, t]],
                  [acc_n[2, t], 0, -acc_n[0, t]],
                  [-acc_n[1, t], acc_n[0, t], 0]])

    # State transition matrix.
    F = np.array([[np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
                  [np.zeros((3, 3)), np.eye(3),  dt*np.eye(3)],
                  [-dt*S, np.zeros((3, 3)), np.eye(3)]]).reshape(9, 9)

    # Compute the process noise covariance Q.
    Q = np.zeros((9, 9), float)
    np.fill_diagonal(Q, [sigma_omega, sigma_omega,
                         sigma_omega, 0, 0, 0, sigma_a, sigma_a, sigma_a])
    Q = (Q*dt)**2

    ft = np.transpose(F)
    # Propagate the error covariance matrix.
    #s = F@P
    print(F)
    #P = F@P@ft + Q


# %%
