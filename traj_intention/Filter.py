import numpy as np
from functools import partial
from scipy import interpolate, signal
import pykalman
from pathlib import Path
import os,sys
FILE_DIR = Path(__file__).resolve().parent
# import pickle
# from Dataset import camera_to_world
# sys.path.append('./depthai_blazepose')
# import mediapipe_utils as mpu
import pdb

class Smooth_Filter(object):
	"""
	Smooth the trajectory data
	"""
	def __init__(self,flag='kalman',kernel_size=3, decay_rate=0.6):
		"""
		:param flag: string, specifies the method for smooth filtering,
				'kalman': kalman filter
				'wiener': weiner filter
				'median': median filter
				'moving_average' or 'ma': moving average filter
				'weighted_moving_average' or 'wma': weighted moving average filter
				'exponential_moving_average' or 'ema': exponential moving average
		:param kernel_size: int, kernel size for median filter or wiener filter or moving average filter
		:param decay_rate: float, decay rate for exponential moving average or weighted moving average filter
		"""
		self.flag = flag
		self.kernel_size = kernel_size
		self.decay_rate = decay_rate
		if self.flag=='median':
			self.filter = partial(self._median_filter,kernel_size=kernel_size)
		elif self.flag=='wiener':
			self.filter = partial(self._wiener_filter, kernel_size=kernel_size)
		elif self.flag=='kalman':
			self.filter = self._kalman_filter
		elif self.flag=='moving_average' or self.flag=='ma':
			self.filter = partial(self._ma_filter, kernel_size=kernel_size)
		elif self.flag=='exponential_moving_average' or self.flag=='ema':
			self.filter = partial(self._ema_filter, decay_rate=decay_rate)
		elif self.flag=='weighted_moving_average' or self.flag=='wma':
			self.filter = partial(self._wma_filter, decay_rate=decay_rate)


		if flag not in self.all_flags:
			raise('invalid  flags. Only support:', self.all_flags)

	def _median_filter(self, trajectory,kernel_size):
		"""
		smooth the  time series data with median filter
		:param trajectory: numpy-array
		:param kernel_size: int,  the size of the median filter window
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			filt_traj[:,ii] = signal.medfilt(trajectory[:,ii], kernel_size=kernel_size)
			filt_traj[:, 0] = filt_traj[:, 1]
			filt_traj[:, -1] = filt_traj[:, -2]
		return filt_traj

	def _wiener_filter(self,trajectory,kernel_size):
		"""
		smooth the  time series data with Wiener filter
		:param trajectory: numpy-array
		:param kernel_size: int,  the size of the Wiener filter window
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			filt_traj[:,ii] = signal.wiener(trajectory[:,ii], mysize=kernel_size)
			filt_traj[:, 0] = filt_traj[:, 1]
			filt_traj[:, -1] = filt_traj[:, -2]
		return filt_traj

	def _kalman_filter(self,trajectory):
		"""
		smooth the  time series data with Kalman filter
		:param trajectory: numpy-array
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]

		self.kf = pykalman.KalmanFilter(n_dim_obs=dim, n_dim_state=dim,initial_state_mean=trajectory[0])
		state_mean, state_covariance = self.kf.filter(trajectory)
		filt_traj = state_mean
		return filt_traj

	def _ma_filter(self,trajectory,kernel_size):
		"""
		smooth the  time series data with moving average
		:param trajectory: numpy-array
		:param kernel_size: int,  the size of the moving average filter window
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		r = np.arange(1, kernel_size - 1, 2)
		for ii in range(dim):
			a = trajectory[:,ii]
			out0 = np.convolve(a, np.ones(kernel_size, dtype=int), 'valid') / kernel_size
			start = np.cumsum(a[:kernel_size - 1])[::2] / r
			stop = (np.cumsum(a[:-kernel_size:-1])[::2] / r)[::-1]
			filt_traj[:,ii] =  np.concatenate((start, out0, stop))
		return filt_traj

	def _ema_filter(self,trajectory,decay_rate):
		"""
		smooth the  time series data with exponential moving average
		:param trajectory: numpy-array
		:param decay_rate: float,  decay rate for exponential moving average
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			a = trajectory[:,ii]
			smoothed = [a[0]]
			for val in a[1:]:
				new_val = decay_rate * val + (1 - decay_rate) * smoothed[-1]
				smoothed.append(new_val)
			filt_traj[:, ii] = np.array(smoothed)
		return filt_traj

	def _wma_filter(self,trajectory,decay_rate):
		"""
		smooth the  time series data with weighted moving average
		:param trajectory: numpy-array
		:param decay_rate: float,  decay rate for weighted moving average
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			a = trajectory[:,ii]
			smoothed = [a[0]]
			for jj in range(1,len(a)):
				new_val = decay_rate * a[jj] + (1 - decay_rate) * a[jj-1]
				smoothed.append(new_val)
			filt_traj[:, ii] = np.array(smoothed)
		return filt_traj

	def smooth_trajectory(self,trajectory):
		"""
		smooth the  time series data
		:param trajectory: numpy array, shape of (time_step,coordinate_dim)
		:return: numpy-array, smoothed time series data, same shape as input series
		"""
		trajectory=np.array(trajectory)
		if len(trajectory.shape)<2:
			trajectory=trajectory.reshape(trajectory.shape[0],1)
		smoothed=self.filter(trajectory)
		return smoothed

	def smooth_multi_trajectories(self,trajectories):
		"""
		smooth the  multi-joint-trajectories  data
		:param trajectories: numpy array, shape of (time_step,joint_num, coordinate_dim)
		:return: numpy-array, smoothed trajectories, same shape as input trajectories
		"""
		trajectories=np.array(trajectories)
		if len(trajectories.shape)<3:
			trajectories=np.expand_dims(trajectories, axis=1)
		joint_num = trajectories.shape[1]
		multi_joint_smoothed = trajectories.copy()
		for ii in range(joint_num):
			multi_joint_smoothed[:,ii] = self.smooth_trajectory(trajectories[:,ii])
		return multi_joint_smoothed

	@property
	def all_flags(self):
		flags=[
			"median",
			"wiener",
			"kalman",
			"moving_average",
			"ma",
			"exponential_moving_average",
			"ema",
			"weighted_moving_average",
			"wma",
		]
		return flags

if __name__ == "__main__":
    smooth_filter = Smooth_Filter(flag="kalman")
    pic = open(f'{FILE_DIR}/../human_traj/abu_connectors/abu_connectors001.pkl','rb')
    pkl_file = pickle.load(pic)
    poses = []
    for body in pkl_file:
        poses.append(body.landmarks)
    npy_file = np.array(poses)
    npy_file = 2*(npy_file-npy_file.min())/(npy_file.max()-npy_file.min())
    npy_file = camera_to_world(npy_file)
    npy_file[:, :, 2] -= np.min(npy_file[:, :, 2])
    smooth_filter.smooth_multi_trajectories(npy_file)
    pdb.set_trace()