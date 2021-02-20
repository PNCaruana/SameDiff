from dataLoader import DataLoader
import sys
import os
import cv2
import numpy as np
from numpy import cos
from numpy import sin
from numpy import pi
from scipy.spatial.transform import Rotation as R


# <a,b,c> -> yaw, pitch, roll -> (zRot, yRot, xRot)
def rotMatrix(a, b, c):
	M = [[cos(a) * cos(b), cos(a) * sin(b) * sin(c) - sin(a) * cos(c), cos(a) * sin(b) * cos(c) + sin(a) * sin(c)],
		 [sin(a) * cos(b), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c)],
		 [-sin(b), cos(b) * sin(c), cos(b) * cos(c)]]
	return M


class Camera:
	def __init__(self):
		self.q = R.from_quat([0, 0, sin(np.pi/4), -sin(np.pi/4)]).as_quat()
		self.cam_x = 0
		self.cam_y = 5
		self.cam_z = 0
		self.focus = (0, 0, 0)
		self.view = np.zeros((500, 500))

	def getParams(self):
		return {"cam_x": self.cam_x,
				"cam_y": self.cam_y,
				"cam_z": self.cam_z,
				"cam_qw": self.q[3],
				"cam_qx": self.q[0],
				"cam_qy": self.q[1],
				"cam_qz": self.q[2]}


class Robot:

	# <int> objNo, either 0 or 1 corresponding to which index key the robot is working with
	def __init__(self, dataLoader, pairNo, objNo):
		self.cam = Camera()
		self.objNo = objNo
		self.pairNo = pairNo
		self.DL = dataLoader
		self.viewCount = 0
		self.cameraPos = {'r': 5, 'phi': 0, 'theta': 0}  # position of camera in spherical coordinates

	# gets the view based on where the camera currently is
	def processCameraView(self):
		print("Acquiring new view...")
		self.cam.view = self.DL.getView(self.pairNo, self.objNo, self.cam.getParams())
		fileName = os.getcwd() + "/views/robot" + str(self.objNo) + "_" + str(self.pairNo) + "_view" + str(
			self.viewCount) + ".jpg"
		cv2.imwrite(fileName, self.cam.view)
		print("Saved view: " + fileName)

		self.viewCount += 1

	def displayView(self):
		cv2.imshow("view", self.cam.view)
		cv2.waitKey(0)

	#orients the camera in spherical coordinates around the camera focus point (default is 0)
	def setCam(self, r, phi, theta):
		# position the camera
		self.cam.cam_x = r * np.cos(theta) * np.sin(phi)
		self.cam.cam_y = r * np.sin(theta) * np.sin(phi)
		self.cam.cam_z = r * np.cos(phi)

		self.cameraPos["r"] = r
		self.cameraPos["theta"] = theta
		self.cameraPos["phi"] = phi



		# set orientation to look at target

		# spherical unit vectors
		phiHat = [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)]
		thetaHat = [-np.sin(theta), np.cos(theta), 0]
		rHat = [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]

		Orientation = R.from_matrix(rotMatrix(theta, phi, 0)).as_quat()
		self.cam.q = Orientation
		print("Camera oriented to", self.cam.getParams())

	def __str__(self):
		pass


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("ERROR - SameDiff.py requires 1 argument <filepath>")
		exit(0)
	file = str(sys.argv[1])
	DL = DataLoader(file)

	robot1 = Robot(DL, 0, 0)
	robot2 = Robot(DL, 0, 1)

	robot1.setCam(5,0,0)
	robot1.processCameraView()
	for i in range(1,5):
		robot1.setCam(5, 0, i*pi/2/4)
		robot1.processCameraView()
	for i in range(1,5):
		robot1.setCam(5, i*pi/2/4, pi/2)
		robot1.processCameraView()

# params = {
#		'cam_x': -0.911,
#		'cam_y': 1.238,
#		'cam_z': -4.1961,
#		'cam_qw': -0.0544,
#		'cam_qx': -0.307,
#		'cam_qy': 0.9355,
#		'cam_qz': 0.16599
#	}
