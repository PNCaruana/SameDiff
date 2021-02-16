from dataLoader import DataLoader
import sys
import cv2

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("ERROR - SameDiff.py requires 1 argument <filepath>")
		exit(0)
	file = str(sys.argv[1])
	DL = DataLoader(file)
	params = {"cam_x":1, "cam_y":1, "cam_z":1, "cam_qw":1, "cam_qx":1, "cam_qy":1, "cam_qz":1}
	view = DL.getView(0,0,params)
	cv2.imshow("view", view)
	cv2.waitKey("0")
