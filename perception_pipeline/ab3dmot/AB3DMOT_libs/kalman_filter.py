# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
	"""
	This class represents the internal state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self, bbox3D, info):
		"""
		Initialises a tracker using initial bounding box.
		"""
		# # define constant velocity model
		# self.kf = KalmanFilter(dim_x=10, dim_z=7)       
		# self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
		#                       [0,1,0,0,0,0,0,0,1,0],
		#                       [0,0,1,0,0,0,0,0,0,1],
		#                       [0,0,0,1,0,0,0,0,0,0],  
		#                       [0,0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0,0],
		#                       [0,0,0,0,0,0,0,1,0,0],
		#                       [0,0,0,0,0,0,0,0,1,0],
		#                       [0,0,0,0,0,0,0,0,0,1]])     

		# self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
		#                       [0,1,0,0,0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0,0]])

		# define constant velocity model for 9 DoF
		# x = 9dof + three velocities, 
		# TODO: are the velocities needed or should we even add angular velocities
		# we empirically found that including the angular velocity
		# does not really improve the performance

		#without velocity
		# self.kf = KalmanFilter(dim_x=9, dim_z=9)   
		# self.kf.F = np.array([[1,0,0,0,0,0,0,0,0],      # state transition matrix
		#                       [0,1,0,0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0],  
		#                       [0,0,0,0,1,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0],
		#                       [0,0,0,0,0,0,0,1,0],
		#                       [0,0,0,0,0,0,0,0,1]])     

		# self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],      # measurement function,
		#                       [0,1,0,0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,1,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0],
		# 					  [0,0,0,0,0,0,0,1,0],
		# 					  [0,0,0,0,0,0,0,0,1]])
		# #with velocity
		self.kf = KalmanFilter(dim_x=12, dim_z=9)   
		self.kf.F = np.array([[1,0,0,0,0,0,0,0,0,1,0,0],      # state transition matrix
		                      [0,1,0,0,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,0,0,1,0,0],
							  [0,0,0,0,0,0,0,0,0,0,1,0],
							  [0,0,0,0,0,0,0,0,0,0,0,1]])     

		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],      # measurement function,
		                      [0,1,0,0,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0,0,0],
							  [0,0,0,0,0,0,0,1,0,0,0,0],
							  [0,0,0,0,0,0,0,0,1,0,0,0]])

		# with angular velocity
		# self.kf = KalmanFilter(dim_x=15, dim_z=9)       
		# self.kf.F = np.array([[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0],      # state transition matrix
		#                       [0,1,0,0,0,0,0,0,0, 0,1,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0, 0,0,1,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0],  
		#                       [0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0, 0,0,0,1,0,0],
		#                       [0,0,0,0,0,0,0,1,0, 0,0,0,0,1,0],
		#                       [0,0,0,0,0,0,0,0,1, 0,0,0,0,0,1],
		#                       [0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0],
		# 					  [0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0],
		# 					  [0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0],
		# 					  [0,0,0,0,0,0,0,0,0, 0,0,0,1,0,0],
		# 					  [0,0,0,0,0,0,0,0,0, 0,0,0,0,1,0],
		# 					  [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1]])     

		# self.kf.H = np.array([[1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0],      # measurement function,
		#                       [0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0],
		# 					  [0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0],
		# 					  [0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0]])
		
		#self.kf.R[0:,0:] *= 10.   # measurement uncertainty # uncomment if no trust in measurement
		self.kf.P[9:, 9:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
		self.kf.P *= 10.

		# self.kf.Q[-1,-1] *= 0.01    # process uncertainty
		self.kf.Q[9:, 9:] *= 0.01
		self.kf.x[:9] = bbox3D.reshape((9, 1))

		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.hits = 1           # number of total hits including the first detection
		self.hit_streak = 1     # number of continuing hit considering the first detection
		self.first_continuing_hit = 1
		self.still_first = True
		self.age = 0
		self.info = info        # other info associated

	def orientation_correction(theta):
		pass
		#theta_corrected = pass

		#return theta_corrected


	def update(self, bbox3D, info): 
		""" 
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1          # number of continuing hit
		if self.still_first:
			self.first_continuing_hit += 1      # number of continuing hit in the fist time

		######################### orientation correction
		self.kf.x[6:9][self.kf.x[6:9] >= np.pi] -= np.pi * 2    # make the theta still in the range
		self.kf.x[6:9][self.kf.x[6:9] < -np.pi] += np.pi * 2

		new_orientations = bbox3D[6:9]
		new_orientations[new_orientations >= np.pi] -= np.pi * 2    # make the theta still in the range
		new_orientations[new_orientations < -np.pi] += np.pi * 2
		bbox3D[6:9] = new_orientations

		predicted_orientations = self.kf.x[6:9]

		# if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
		# 	self.kf.x[3] += np.pi       
		# 	if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
		# 	if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
		
		# numerical trick only makes sense if only yaw angle and parallel boxes are considered
		# rotation of 90 degree of a box leads to the same box

		# if the angle of two theta is not acute angle

		for i in [6,7,8]:
			diff = new_orientations[i-6] - predicted_orientations[i-6]
			check = abs(diff) > np.pi # / 2.0 and abs(diff) < np.pi * 3 / 2.0
			# print(check)
			if check: 
				if predicted_orientations[i-6] < 0:
					self.kf.x[i] += 2 * np.pi
				else:
					self.kf.x[i] -= 2 * np.pi

		# diff = new_orientations.reshape((3,1)) - predicted_orientations
		# check_1 = abs(diff) > np.pi / 2.0
		# #print(check_1)
		# check_2 = abs(diff) < np.pi * 3 / 2.0
		# print(np.logical_and(check_1, check_2))

		# print(diff)
		# if np.logical_or(diff < 0):
		# 	self.kf.x[6:9][np.logical_and(check_1, check_2)] -= 2 * np.pi
		# else:
		# 	self.kf.x[6:9][np.logical_and(check_1, check_2)] += 2 * np.pi
		
		# self.kf.x[6:9][self.kf.x[6:9] >= np.pi] -= np.pi * 2    # make the theta still in the range
		# self.kf.x[6:9][self.kf.x[6:9] < -np.pi] += np.pi * 2

		### print(self.kf.x[6:9])
		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		# for i in [6,7,8]:
		# 	if abs(new_orientations[i-6] - self.kf.x[i]) >= np.pi * 3 / 2.0:
		# 		print("---------------------------------------")
		# 		print("---------------------------------------")
		# 		print("---------------------------------------")
		# 		if new_orientations[i-6] > 0: 
		# 			self.kf.x[i] += np.pi * 2
		# 		else: 
		# 			self.kf.x[i] -= np.pi * 2
		# 		print(self.kf.x[6:9])
		# 		print(bbox3D[6:9])

		#########################     # flip

		self.kf.update(bbox3D)

		# print("After Update")
		# print(self.kf.x[6:9])
		# print(bbox3D[6:9])

		self.kf.x[6:9][self.kf.x[6:9] >= np.pi] -= np.pi * 2    # make the theta still in the range
		self.kf.x[6:9][self.kf.x[6:9] < -np.pi] += np.pi * 2
		self.info = info

		# print("New state: ", self.kf.x[6:9])

	def predict(self):       
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		self.kf.predict()      
		self.kf.x[6:9][self.kf.x[6:9] >= np.pi] -= np.pi * 2    # make the theta still in the range
		self.kf.x[6:9][self.kf.x[6:9] < -np.pi] += np.pi * 2

		self.age += 1
		if (self.time_since_update > 0):
			self.hit_streak = 0
			self.still_first = False
		self.time_since_update += 1
		self.history.append(self.kf.x)
		return self.history[-1]

	def get_state(self):
		"""
		Returns the current bounding box estimate.
		"""
		return self.kf.x[:9].reshape((9, ))