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
		self.offset = [0,0,0]

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

	def showImg(self, img):
		cv2.imshow('img', img)
		cv2.waitKey(0)

	def getCentroid(self):
		#get contours of view
		ret, thresh = cv2.threshold(self.cam.view, 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(self.cam.view, contours, -1, (0,255,0), 3)


		#find centroid
		for c in contours:
			M = cv2.moments(c)

			# calculate x,y coordinate of center
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])

			cv2.circle(self.cam.view, (cX, cY), 5, (0, 255, 0), -1)

		if not contours:
			print("ERROR -> No object in view")
			return -1
		else:
			cv2.imwrite("views/contours_"+str(self.viewCount - 1)+".jpg", self.cam.view)
			return (cX, cY)


	def centerObject(self, debug=False):
		self.setCam(4, 0, 0) # looking straight down
		if not debug:
			self.processCameraView()
		else:
			self.cam.view = cv2.imread("views/robot0_0_view0.jpg", 0) #for debug
			self.viewCount = 1

		#get distance
		C1 = self.getCentroid()
		self.offset[0] = 0.1
		self.setCam(4, 0, 0)

		if not debug:
			self.processCameraView()
		else:
			self.cam.view = cv2.imread("views/robot0_0_view1.jpg", 0)  # for debug
			self.viewCount = 2

		C2 = self.getCentroid()

		#stereo-distance:
		f = 3500 #35 mm
		b = 0.1
		Z = (b*f)/(np.abs(C1[0] - C2[0])) #distance in pixels of center along x-axis
		print("Distance to camera is: " + str(Z))
		print("Centroid_0:" + str(C1))

		dp = (960 - C1[0]) # distance from center when camera is at (0,0,R) in pixels for 1920x1080 image
		dx = Z*dp/f #actual distance

		dp = (540 - C1[1])
		dy = Z*dp/f

		print("x-distance: " + str(dx))
		print("y-distance: " + str(dy))
		self.offset[0] = dx #move camera so that centroid is centered
		self.offset[1] = dy
		self.setCam(4,0,0)

		if not debug:
			self.processCameraView()
		else:
			self.cam.view = cv2.imread("views/robot0_0_view2.jpg", 0)  # for debug
			self.viewCount = 3

		CF = self.getCentroid()
		print("Sanity Check: " + str(CF))





	#orients the camera in spherical coordinates around the camera focus point (default is 0)
	def setCam(self, r, phi, theta):
		# position the camera
		self.cam.cam_x = r * np.cos(theta) * np.sin(phi) + self.offset[0]
		self.cam.cam_y = r * np.sin(theta) * np.sin(phi) + self.offset[1]
		self.cam.cam_z = r * np.cos(phi) + self.offset[2]

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

	robot1.centerObject(debug = True)

# params = {
#		'cam_x': -0.911,
#		'cam_y': 1.238,
#		'cam_z': -4.1961,
#		'cam_qw': -0.0544,
#		'cam_qx': -0.307,
#		'cam_qy': 0.9355,
#		'cam_qz': 0.16599
#	}
