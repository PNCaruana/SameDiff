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

def normalize(a):
    a = np.array(a)
    n = 0
    for x in a:
        n += x**2
    N = np.sqrt(n)
    return a/N

def orientation(right, up, forward):
    right = normalize(right)
    up = normalize(up)
    forward = normalize(forward)

    M = [[right[0], up[0], forward[0]],
         [right[1], up[1], forward[1]],
         [right[2], up[2], forward[2]]]
    return M

class Camera:
    def __init__(self):
        self.q = R.from_quat([0, 0, sin(np.pi / 4), -sin(np.pi / 4)]).as_quat()
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

    def printParams(self):
        return str(self.cam_x) + "," + str(self.cam_y) + "," + str(self.cam_z) + "," + str(self.q[3]) + "," + str(
            self.q[0]) + "," + str(self.q[1]) + "," + str(self.q[2]) + "\n"

    #target is Vector3 <x,y,z>
    def lookAt(self, target):
        up = np.array([0.00000000000000001, 0, 1]) # slightly peturbed to avoid singularity
        pos = np.array([self.cam_x, self.cam_y, self.cam_z])
        target = np.array(target)

        f = (target - pos); f = f/np.linalg.norm(f)
        s = np.cross(f, up); s = s/np.linalg.norm(s)
        u = np.cross(s, f); u = u/np.linalg.norm(u)

        rot = np.zeros((3, 3))

        rot[0:3, 0] = s
        rot[0:3, 1] = u
        rot[0:3, 2] = -f

        Orientation = R.from_matrix(rot).as_quat()
        self.q = np.round(Orientation, 8)




class Robot:

    # <int> objNo, either 0 or 1 corresponding to which index key the robot is working with
    def __init__(self, dataLoader, pairNo, objNo, debug=False):
        self.cam = Camera()
        self.objNo = objNo
        self.pairNo = pairNo
        self.DL = dataLoader
        self.viewCount = 0
        self.cameraPos = {'r': 4, 'phi': pi/2, 'theta': 0}  # position of camera in spherical coordinates
        self.defaultCam = {'r': 4, 'phi': pi/2, 'theta': 0}
        self.moveSpeed = pi/8
        self.offset = [0, 0, 0]
        self.debugMode = debug


    # gets the view based on where the camera currently is
    def processCameraView(self, debug=False):

        if not debug:

            print("Acquiring new view...")
            self.cam.view = self.DL.getView(self.pairNo, self.objNo, self.cam.getParams())
            fileName = os.getcwd() + "/views/robot" + str(self.objNo) + "_" + str(self.pairNo) + "_view" + str(
                self.viewCount) + ".jpg"
            cv2.imwrite(fileName, self.cam.view)

            if self.debugMode:
                print("Saved view: " + fileName)

        self.viewCount += 1

    def displayView(self):
        cv2.imshow("view", self.cam.view)
        cv2.waitKey(0)

    def showImg(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)

    # gets centroid of image blob
    def getCentroid(self):
        # get contours of view
        ret, thresh = cv2.threshold(self.cam.view, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.cam.view, contours, -1, (0, 255, 0), 3)

        # find centroid
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
            cv2.imwrite("views/contours_" + str(self.viewCount - 1) + ".jpg", self.cam.view)
            return (cX, cY)

    # centers object in world coordinates
    def centerObject(self, debug=False):
        print("Stereoscopically centering object...")
        print("  >Getting first view")
        if not debug:
            self.setCam(4, 0, 0)
            self.processCameraView()
        else:
            self.cam.view = cv2.imread("views/robot0_0_view0.jpg", 0)  # for debug
            self.setCam(4, 0, 0)
            self.viewCount = 1

        # get distance
        C1 = self.getCentroid()
        self.offset[1] = 0.2

        print("  >Getting second view from different position (offset 0.2)")
        if not debug:
            self.setCam(4, 0, 0)
            self.processCameraView()
        else:
            self.cam.view = cv2.imread("views/robot0_0_view1.jpg", 0)  # for debug
            self.setCam(4, 0, 0)
            self.viewCount = 2

        C2 = self.getCentroid()

        # stereo-distance:
        f = 2110  # apparent focus wtf
        b = 0.2
        Z = (b * f) / (np.abs(C1[0] - C2[0]))  # distance in pixels of center along x-axis

        dp = (960 - C1[0])  # distance from center when camera is at (0,0,R) in pixels for 1920x1080 image
        dy = Z * dp / f  # actual distance

        dp = (540 - C1[1])
        dx = Z * dp / f
        dz = 4 - Z  # camera z-pos minus depth
        print("  >calculated distance to camera is: " + str(round(Z, 4)))

        self.offset[0:3] = np.round([dx, dy, dz], 8)

        if not debug:
            self.setCam(4, 0, 0)
            self.processCameraView()
        else:
            self.cam.view = cv2.imread("views/robot0_0_view2.jpg", 0)  # for debug
            self.setCam(4, 0, 0)
            self.viewCount = 3

        CF = self.getCentroid()

        print("  >final object centroid at " + str((dx, dy, dz)))
        print("Object Centered")

        self.cameraPos = self.defaultCam

    # interface functions ==========================================

    # moves camera in directions specified by fn name
    def move_down(self):
        newPhi = self.cameraPos["phi"] - self.moveSpeed
        if 0 <= newPhi:
            self.cameraPos["phi"] = newPhi
            print("moved DOWN to " + str(self.cameraPos))

            if round(newPhi, 8) == 0:
                newPhi = 0.00001 # dont want it exactly at singularity
                self.cameraPos["phi"] = newPhi

        else:
            print("Cannot move UP, already at maximum")
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def move_up(self):
        newPhi = self.cameraPos["phi"] + self.moveSpeed
        if newPhi <= pi:
            self.cameraPos["phi"] = newPhi
            print("moved UP to " + str(self.cameraPos)) # tell the folks at home

            if round(newPhi, 8) == pi:
                newPhi = pi - 0.00001 # dont want it exactly at singularity
                self.cameraPos["phi"] = newPhi
        else:
            print("Cannot move DOWN, already at minimum")
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def move_left(self):
        newTheta = self.cameraPos["theta"] - self.moveSpeed
        self.cameraPos["theta"] = newTheta % (2*pi)
        print("moved LEFT to " + str(self.cameraPos))
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def move_right(self):
        newTheta = self.cameraPos["theta"] + self.moveSpeed
        self.cameraPos["theta"] = newTheta % (2*pi)
        print("moved RIGHT to " + str(self.cameraPos))
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def set_radius(self, r):
        self.cameraPos["r"] = r
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])
    # orients the camera in spherical coordinates around the camera focus point (default is 0)
    def setCam(self, r, phi, theta):
        # position the camera
        self.cam.cam_x = np.round(r * np.cos(theta) * np.sin(phi) + self.offset[0], 8)
        self.cam.cam_y = np.round(r * np.sin(theta) * np.sin(phi) + self.offset[1], 8)
        self.cam.cam_z = np.round(r * np.cos(phi) + self.offset[2], 8)

        self.cameraPos["r"] = r
        self.cameraPos["theta"] = theta
        self.cameraPos["phi"] = phi

        # set orientation to look at target

        # spherical unit vectors
        phiHat = np.round([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)],8)
        thetaHat = np.round([-np.sin(theta), np.cos(theta), 0],8)
        rHat = np.round([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],8)

        # point to the center of the view sphere
        self.cam.lookAt(self.offset)


        if self.debugMode:
            print("Camera oriented to", self.cam.getParams())
            # save transcript of orientations and positions
            fileName = os.getcwd() + "/views/movement.txt"
            f = open(fileName, 'a')
            f.write(self.cam.printParams())
            f.close()

    def __str__(self):
        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR - SameDiff.py requires 1 argument <filepath>")
        exit(0)

    if os.path.exists("views/movement.txt"):
        os.remove("views/movement.txt")

    file = str(sys.argv[1])
    DL = DataLoader(file)

    robot1 = Robot(DL, 1, 0, debug=False)
    robot2 = Robot(DL, 1, 1)

    robot1.centerObject(debug=False)

    robot1.set_radius(1.5)

    for i in range (0, 6):
        robot1.move_left()
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
