from dataLoader import DataLoader
import sys
import cv2

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("ERROR - SameDiff.py requires 1 argument <filepath>")
		exit(0)
	file = str(sys.argv[1])
	DL = DataLoader(file)
	params = {
		'cam_x': -0.911,
		'cam_y': 1.238,
		'cam_z': -4.1961,
		'cam_qw': -0.0544,
		'cam_qx': -0.307,
		'cam_qy': 0.9355,
		'cam_qz': 0.16599
	}
	view = DL.getView(0, 0, params, isRandomCam="true")
	cv2.imshow("view", view)
	cv2.waitKey(0)
