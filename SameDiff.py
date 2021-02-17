from dataLoader import DataLoader
import sys
import os
import cv2
import numpy as np
import quaternion


class Camera:
	def __init__(self):
		self.q = np.quaternion(0, 0, 0.707, 0.707)
		self.cam_x = 0
		self.cam_y = 5
		self.cam_z = 0

		self.view = np.zeros((500,500))

	def getParams(self):
		return {"cam_x":self.cam_x,
				"cam_y":self.cam_y,
				"cam_z":self.cam_z,
				"cam_qw":self.q.w,
				"cam_qx":self.q.x,
				"cam_qy":self.q.y,
				"cam_qz":self.q.z}

class Robot:

	# <int> objNo, either 0 or 1 corresponding to which index key the robot is working with
	def __init__(self, dataLoader, pairNo, objNo):
		self.cam = Camera()
		self.objNo = objNo
		self.pairNo = pairNo
		self.DL = dataLoader
		self.viewCount = 0
		self.cameraPos = {'r': 5, 'phi': 0, 'theta': 0} #position of camera in spherical coordinates

	#gets the view based on where the camera currently is
	def processCameraView(self):
		self.cam.view = self.DL.getView(self.pairNo, self.objNo, self.cam.getParams())
		fileName = os.getcwd()+"/views/robot"+str(self.objNo)+"_"+str(self.pairNo)+"_view" + str(self.viewCount)+".jpg"
		cv2.imwrite(fileName, self.cam.view)
		print("Saved view: " + fileName)

		self.viewCount += 1

	def displayView(self):
		cv2.imshow("view", self.cam.view)
		cv2.waitKey(0)

	def setCam(self, r, phi, theta):
		#position the camera
		self.cam.cam_x = r*np.cos(theta)*np.sin(phi)
		self.cam.cam_y = r*np.sin(theta)*np.sin(phi)
		self.cam.cam_z = r*np.cos(phi)

		print(self.cam.getParams())

		#set orientation to look at target

	def __str__(self):
		pass



if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("ERROR - SameDiff.py requires 1 argument <filepath>")
		exit(0)
	file = str(sys.argv[1])
	DL = DataLoader(file)

	robot1 = Robot(DL, 0, 0)

	for i in range(0,3):
		robot1.setCam(5, np.pi/2, np.pi/2 +i*0.2)
		robot1.processCameraView()


	#params = {
#		'cam_x': -0.911,
#		'cam_y': 1.238,
#		'cam_z': -4.1961,
#		'cam_qw': -0.0544,
#		'cam_qx': -0.307,
#		'cam_qy': 0.9355,
#		'cam_qz': 0.16599
#	}


