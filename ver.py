# %%
import sympy as sp
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
df = pd.read_csv('ver_data.csv', encoding='unicode_escape')
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

g = 9.8  # Gravity.

# divert acc_s to unit to m/s2
acc_s = acc_s.to_numpy() * g
acc_s.shape

gyro_s = df.loc[:, 'gyr_x': 'gyr_z']  # Rates of turn in sensor frame.
gyro_s

# Initialise parameters.
# Orientation from accelerometers. Sensor is assumed to be stationary.

pitch = -asin(acc_s[0, 0]/g)
pitch
roll = atan(acc_s[0, 1]/acc_s[0, 2])
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

# Gyroscope bias, to be determined for each sensor.
# Defined above so we don't forget to change for each dataset. --
# Preallocate storage for accelerations in navigation frame.


acc_n = np.empty((3, data_size))
acc_n[:] = np.nan
acc_n
acc_n.shape

# Convert the first columns from df to np.values
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
distance = np.empty((1, data_size))
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

for t in range(1, data_size):
    # Start INS (transformation, double integration)
    dt = timestamp[t] - timestamp[t-1]

    # Remove bias from gyro measurements.
    gyro_s1 = (gyro_s[t: t+1] - gyro_bias).to_numpy()

    # Skew-symmetric matrix for angular rates
    ang_rate_matrix = np.array([[0, -gyro_s1[0, 2], gyro_s1[0, 1]],
                                [gyro_s1[0, 2], 0, -gyro_s1[0, 0]],
                                [-gyro_s1[0, 1], gyro_s1[0, 0], 0]])

    # orientation esimation ** chrck point **
    C = C_prev @ (2 * np.eye(3) + (ang_rate_matrix * dt)
                  ) @ np.linalg.inv(2*np.eye(3)-(ang_rate_matrix*dt))

    # Transforming the acceleration from sensor frame to navigation frame.
    acc_n[:, t] = 0.5*(C + C_prev)@acc_s[:, t]
    acc_n[:, t]

    # Velocity and position estimation using trapeze integration.
    vel_n[:, t] = vel_n[:, t-1] + \
        ((acc_n[:, t] - [0, 0, g])+(acc_n[:, t-1] - [0, 0, g]))*dt/2

    pos_n[:, t] = pos_n[:, t-1] + (vel_n[:, t] + vel_n[:, t-1])*dt/2

    # Skew-symmetric cross-product operator matrix formed from the n-frame accelerations.
    S = np.array([[0, -acc_n[2, t], acc_n[1, t]],
                  [acc_n[2, t], 0, -acc_n[0, t]],
                  [-acc_n[1, t], acc_n[0, t], 0]])

    # State transition matrix.
    F = np.block([[np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
                  [np.zeros((3, 3)), np.eye(3),  dt*np.eye(3)],
                  [-dt*S, np.zeros((3, 3)), np.eye(3)]])

    # Compute the process noise covariance Q.
    Q = np.zeros((9, 9), float)
    np.fill_diagonal(Q, [sigma_omega, sigma_omega,
                         sigma_omega, 0, 0, 0, sigma_a, sigma_a, sigma_a])
    Q = (Q*dt)**2

    # Propagate the error covariance matrix.
    P = F @ P @ F.T + Q

    ######## End INS ########

    # Stance phase detection and zero-velocity updates.
    if np.linalg.norm(gyro_s[t:t+1]) < gyro_threshold:
        ## Start Kalman filter zero-velocity update ##
        # Kalman gain./((H@P@H.T) + R) (P@H.T)
        K = (P @ H.T) @ np.linalg.inv(((H @ P @ H.T) + R))

        # Update the filter state.
        delta_x = K @ vel_n[:, t]

        # Update the error covariance matrix.
        # Joseph form to guarantee symmetry and positive-definiteness.
        P = (np.eye(9) - K @ H) @ P @ (np.eye(9) - K @ H).T + K @ R @ K.T
        # Simplified covariance update found in most books.
        P = (np.eye(9) - K @ H) @ P

        # Extract errors from the KF state.
        attitude_error = delta_x[0:3]
        pos_error = delta_x[3:6]
        vel_error = delta_x[6:9]
        attitude_error = delta_x[0:3]

        #### End Kalman filter zero-velocity update ###

        # Apply corrections to INS estimates. %%%
        # Skew-symmetric matrix for small angles to correct orientation.
        ang_matrix = -np.array([[0, -attitude_error[2], attitude_error[1]],
                                [attitude_error[2], 0, -attitude_error[0]],
                                [-attitude_error[1], attitude_error[0], 0]])

        # Correct orientation.
        C = (2 * np.eye(3) + (ang_matrix)
             ) @ np.linalg.inv(2 * np.eye(3)-(ang_matrix)) @ C

        # Correct position and velocity based on Kalman error estimates.
        vel_n[:, t] = vel_n[:, t] - vel_error
        pos_n[:, t] = pos_n[:, t] - pos_error

    # Estimate and save the yaw of the sensor (different from the direction of travel). Unused here but potentially useful for orienting a GUI correctly.
    heading[0, t] = atan2(C[1, 0], C[0, 0])
    C_prev = C  # Save orientation estimate, required at start of main loop.

    # Compute horizontal distance.
# distance(1,t) = distance(1,t-1) + sqrt((pos_n(1,t)-pos_n(1,t-1))^2 + (pos_n(2,t)-pos_n(2,t-1))^2);
    distance[0, t] = distance[0, t-1] + \
        sqrt((pos_n[0, t]-pos_n[0, t-1]) ** 2 +
             (pos_n[1, t] - pos_n[1, t-1]) ** 2)


# Rotate position estimates and plot.
# Rotation angle required to achieve an aesthetic alignment of the figure.

rotation_matrix = np.array(
    [[np.cos(np.pi).astype(int), -np.sin(np.pi).astype(int)],
     [np.sin(np.pi).astype(int), np.cos(np.pi).astype(int)]])

rotation_matrix

pos_r = np.zeros((2, data_size))
for idx in range(0, data_size):
    pos_r[:,
          idx] = rotation_matrix @ np.array([pos_n[0, idx], pos_n[1, idx]]).T

# plt(pos_r[0, :], pos_r[1, :], c='red', lw=2)
# start = plt(pos_r[0, 0], pos_r[1, 0], 'Marker', '^',
#             'LineWidth', 2, 'LineStyle', 'none')
# stop = plt(pos_r[0, -1], pos_r[2, -1], 'Marker',
#            'o', 'LineWidth', 2, 'LineStyle', 'none')


fig1, ax1 = plt.subplots(figsize=(10, 10))
fig1.suptitle('2D Pedestrian Trajectory ', fontsize=20)
ax1.set_xlabel('X position (m)')
ax1.set_ylabel('Y position (m)')
ax1.plot(pos_r[0, :], pos_r[1, :], 'r', lw=2)
ax1.legend(fontsize=15)
ax1.annotate('Start', xy=(pos_r[0, 0], pos_r[1, 0]),  xycoords='data',
             xytext=(-90, -90), textcoords='offset points',
             size=30, ha='right', va="center",
             bbox=dict(boxstyle="round", alpha=0.1), arrowprops=dict(arrowstyle="wedge,tail_width=1", alpha=0.1))

ax1.annotate('End', xy=(pos_r[0, -1], pos_r[1, -1]),  xycoords='data',
             xytext=(90, 90), textcoords='offset points',
             size=30, ha='right', va="center",
             bbox=dict(boxstyle="round", alpha=0.1),
             arrowprops=dict(arrowstyle="wedge,tail_width=1", alpha=0.1))
plt.show


# %%
