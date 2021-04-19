from dataLoader import DataLoader
from playsound import playsound
import sys
import os
import cv2
import numpy as np
from numpy import cos
from numpy import sin
from numpy import pi
from scipy.spatial.transform import Rotation as R
import open3d


#colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
COLORS = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]

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
        n += x ** 2
    N = np.sqrt(n)
    if N == 0:
        return (0,0,0)
    return a / N

def vecMag(a):
    M=0
    for x in a:
       M += x**2
    return np.sqrt(M)

def orientation(right, up, forward):
    right = normalize(right)
    up = normalize(up)
    forward = normalize(forward)

    M = [[right[0], up[0], forward[0]],
         [right[1], up[1], forward[1]],
         [right[2], up[2], forward[2]]]
    return M

def dist_2d(A, B):
    AB = [B[0] - A[0], B[1] - A[1]]
    return round(np.sqrt(AB[0] ** 2 + AB[1] ** 2), 8)

def dist_3d(A, B):
    AB = [B[0] - A[0], B[1] - A[1], B[2], A[2]]
    return round(np.sqrt(AB[0] ** 2 + AB[1] ** 2 + AB[2]**2), 8)

def same_2d(A, B):
    if A[0] == B[0] and A[1] == B[1]:
        return True
    return False

# Weighted Least Squares for 2D points centroids
def WLSQ(points):
    #img = np.zeros((1000, 1000, 3), np.uint8)

    points = np.array(points)

    #for V in points:
       # cv2.circle(img, (V[1], V[0]), 3, (0, 0, 255), -1)

    # init weights
    weights = []
    for p in points:
        weights.append(0)

    # get center
    N = len(points)
    avg = np.array([0, 0])
    for p in points:
        avg += np.array(p)
    COM = np.array(avg) // N  # avg is the center of mass
    #cv2.circle(img, (COM[1], COM[0]), 3, (255, 255, 0), -1)

    for i in range(0, len(points)):
        X = points[i] - COM + normalize(
            (np.array([np.random.randint(1, 100), np.random.randint(1, 100)]))) / 10000000000000  # peturbation term
        sq = (X[0] ** 2 + X[1] ** 2)  # singularity term
        weights[i] = (1 / sq)

    avg = np.array([0.00000000001, 0.00000000001], )
    # weighted least squares minimization
    W = np.sum(weights)
    for i in range(0, len(points)):
        avg = avg + (points[i] * weights[i] / W)

    return avg.astype(int)

def WLSQ_3D(points):
    points = np.array(points)

    # init weights
    weights = []
    for p in points:
        weights.append(0)

    # get center
    N = len(points)
    avg = np.array([0., 0., 0.])
    for p in points:
        avg += np.array(p)
    COM = np.array(avg) / N  # avg is the center of mass
    #cv2.circle(img, (COM[1], COM[0]), 3, (255, 255, 0), -1)

    for i in range(0, len(points)):
        X = points[i] - COM + normalize(
            (np.array([np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)]))) / 10000000000000  # peturbation term to avoid singularity
        sq = (X[0] ** 2 + X[1] ** 2 + X[2] ** 2)  # singularity term
        weights[i] = (1 / sq)

    avg = np.array([0.00000000001, 0.00000000001, 0.00000000001], )
    # weighted least squares minimization
    W = np.sum(weights)
    for i in range(0, len(points)):
        avg = avg + (points[i] * weights[i] / W)

    return avg

def fixArrayForFlow(points):
    newP = []
    for p in points:
        newP.append([p[::-1]])
    points = np.array(newP, dtype=np.float32)
    return points

def unfuckFlowArray(points):
    print("Un-fucking optical flow array")
    newP = []
    points = (points.astype(int)).tolist()
    for P in points:
        for p in P:
            newP.append(p)
    return newP

def showImg(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#flattens 3D points and returns an image
def imageFrom3D(pts):
    pass
    pts = np.array(pts)
    pts[:,2] = 0 #remove z values
    minx = np.min(pts[:,0])
    miny = np.min(pts[:,1])
    pts[:, 0] += minx
    pts[:, 1] += miny
    maxx = np.max(pts[:,0])
    maxy = np.max(pts[:,1])
    max = np.max([maxx, maxy])
    pts[:, 0] /= max
    pts[:, 1] /= max

    pts[:, 0] *= 500
    pts[:, 1] *= 500

    pts[:, 0] += 100
    pts[:, 1] += 100

    pts = pts.astype(int)

    img = np.zeros((700, 700), np.int8)
    img[:,:] = 0
    for pt in pts:
        cv2.circle(img, (pt[1], pt[0]), 2, 255, -1)

    #showImg(img)




class Camera:
    def __init__(self):
        self.q = R.from_quat([0, 0, sin(np.pi / 4), -sin(np.pi / 4)]).as_quat()
        self.cam_x = 0
        self.cam_y = 5
        self.cam_z = 0
        self.focus = (0, 0, 0)
        self.view = np.zeros((500, 500))

        self.forward = [0, 0, -1]
        self.up = [1, 0, 0]
        self.side = [0, 1, 0]

        self.rotMat = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]

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

    # target is Vector3 <x,y,z>
    def lookAt(self, target):
        up = np.array([0.00000000000000001, 0, 1])  # slightly peturbed to avoid singularity
        pos = np.array([self.cam_x, self.cam_y, self.cam_z])
        target = np.array(target)

        f = (target - pos);
        f = f / np.linalg.norm(f)
        s = np.cross(f, up);
        s = s / np.linalg.norm(s)
        u = np.cross(s, f);
        u = u / np.linalg.norm(u)

        self.forward = np.round(f, 8)
        self.up = np.round(u, 8)
        self.side = np.round(s, 8)

        rot = np.zeros((3, 3))

        rot[0:3, 0] = s
        rot[0:3, 1] = u
        rot[0:3, 2] = -f

        self.rotMat = rot
        Orientation = R.from_matrix(rot).as_quat()
        self.q = np.round(Orientation, 8)

    # todo: Transforms point from camera coordinates to world coordinates


    def world2cam(self, point):
        point = np.array(point)
        R = [self.side, self.up, self.forward]
        return np.matmul(R, point)

class Face:
    def __init__(self, numPoints, area, sideLengths, pts):
        self.numPoints = numPoints
        self.area = area
        self.sideLengths = sideLengths
        self.pts = pts

        N = len(pts)
        avg = np.array([0., 0., 0.])
        for p in pts:
            avg += np.array(p)
        self.COM = np.array(avg) / N  # avg is the center of mass

        A = np.array(pts[0])
        B = np.array(pts[1])
        C = np.array(pts[2])
        AB = B-A
        AC = C-A
        N = np.cross(AB, AC)
        N = np.array(normalize(N))
        #We want normal to point away from origin
        OC = self.COM  # COM - O = COM
        dot = np.dot(OC, N)
        add = OC + N
        add = vecMag(add)
        if not((dot >=0) and (add >= vecMag(OC))):
            N = -N

        self.normal = np.array(normalize(N))


    #Averages everything with other face
    def mergeWithFace(self, face):
        self.area = 0.5*(self.area + face.area)
        self.normal = (self.normal + face.normal)/2

    def compareToFace(self, other):
        areaQuot = np.round(self.area/other.area, 8)

class Polyhedron:


    def __init__(self):
        self.points = []  # list of all points in 3D
        self.colors = []
        self.faces = []  # list of groups of points which make up faces
        self.pointCloud = open3d.geometry.PointCloud()
        self.boundingVolume = 0

    def addFace(self, fcs):
        newFaces = []
        for face in fcs:
            np = len(face)
            area = self.polygonArea(face)
            sideLengths = self.sideLengths(face)
            newFaces.append(Face(np, area, sideLengths, pts=face))

        self.faces.extend(newFaces)
        #self.pruneFaces()

    def polygonArea(self, pts):
        sum = 0
        for i in range(0, len(pts)):
            for j in range(0, len(pts)):
                if i != j:
                    p1 = pts[i]
                    p2 = pts[j]
                    sum += (p1[0]*p2[1] - p1[1]*p2[0])

        sum = sum/2
        return np.abs(sum)

    def pruneFaces(self):
        print("Pruning Faces")
        sameExists = True
        while sameExists:
            print(" ... Still Pruning")
            same = False
            for i in range(0, len(self.faces) - 1):
                face_i = self.faces[i]
                for j in range(i+1, len(self.faces)):
                    face_j = self.faces[j]
                    # First, compare area
                    if face_i.area * 0.95 <= face_j.area <= face_i.area * 1.05:  # within +- 20% difference
                        #print("face i norm: " + str(face_i.normal))
                        #print("face j norm: " + str(face_j.normal))
                        #print("abs dot: " + str(np.abs(np.dot(face_i.normal, face_j.normal))))
                        if np.abs(np.dot(face_i.normal, face_j.normal)) > 0.7: #normals aligned
                            same = True
                            break

                if same:
                    face_i.mergeWithFace(face_j)
                    self.faces.pop(j)
                    break
            if same:
                continue
            sameExists = False  # We only reach here if no more similar points are found
        self.sortFaces() #Sort faces by area when done

    # Sorts faces in polygon by area, Descending
    def sortFaces(self):
        self.faces.sort(key=lambda x: x.area, reverse=True)

    def comparePoly(self, other):
        numFaces1 = self.numFaces()
        numFaces2 = other.numFaces()

        sa1 = 0 #surface area
        sa2 = 0
        sameNumFaces = numFaces1 == numFaces2
        if sameNumFaces:
            for face in self.faces:
                sa1 += face.area
            sa1 /= numFaces1

            for face in other.faces:
                sa2 += face.area
            sa2 /= numFaces2
        else: #Just compare their biggest faces if they dont have equal amount
            limit = np.min([numFaces1, numFaces2])
            for i in range(0, limit):
                sa1 += self.faces[i].area
                sa2 += other.faces[i].area
            sa1 /= limit
            sa2 /= limit

        return [['SameNumFaces', sameNumFaces],
                ['surfaceArea1', sa1],
                ['surfaceArea2', sa2]]

    def sideLengths(self, pts):
        SL = []
        for i in range(0, len(pts)):
            for j in range(0, len(pts)):
                if i != j:
                    p1 = np.array(pts[i])
                    p2 = np.array(pts[j])
                    p12 = p2-p1
                    M = vecMag(p12)
                    SL.append(M)
        SL.sort()
        return SL

    def calcBoundingVolume(self):
        minxyz = [9999,9999, 9999]
        maxxyz = [-9999, -9999, -9999]
        for p in self.points:
            x = p[0]
            y = p[1]
            z = p[2]
            if x > maxxyz[0]:
                maxxyz[0] = x
            elif x < minxyz[0]:
                minxyz[0] = x

            if y > maxxyz[1]:
                maxxyz[1] = y
            elif y < minxyz[1]:
                minxyz[1] = y

            if z > maxxyz[2]:
                maxxyz[2] = z
            elif z < minxyz[2]:
                minxyz[2] = z

            L = np.abs(maxxyz[0] - minxyz[0])
            W = np.abs(maxxyz[1] - minxyz[1])
            H = np.abs(maxxyz[2] - minxyz[2])

            self.boundingVolume = L*W*H

    def addPoints(self, pts, color=[1, 0, 0]):
        self.points.extend(pts)
        for i in range(0, len(pts)):
            self.colors.append(color)

    #Adds points to polygon and draws lines between all points
    def drawLine(self, pts, color):
        self.addPoints(pts, color)
        lineColor = np.array(color)/2
        for i in range(0, len(pts)):
            for j in range(0, len(pts)):
                if i != j:
                    line = []
                    p1 = np.array(pts[i])
                    p2 = np.array(pts[j])
                    for k in range(1, 50):
                        p12 = p2 - p1
                        lp = p1 + (k/50)*p12
                        line.append(lp)

                    self.addPoints(line, lineColor)

    def drawFaces(self):
        self.points = []
        self.colors = []
        for face in self.faces:
            self.drawLine(face.pts, COLORS.pop(0))

    def numFaces(self):
        return len(self.faces)

    def numPoints(self):
        return len(self.points)

    def viewPoints(self):
        self.drawLine([[0,0,0],[1,0,0]], [1,0,0])
        self.drawLine([[0,0,0],[0,1,0]], [0,1,0])
        self.drawLine([[0,0,0],[0,0,1]], [0,0,1])


        self.pointCloud.points = open3d.utility.Vector3dVector(self.points)
        self.pointCloud.colors = open3d.utility.Vector3dVector(self.colors)
        open3d.visualization.draw_geometries([self.pointCloud])

class Robot:
    # <int> objNo, either 0 or 1 corresponding to which index key the robot is working with
    def __init__(self, dataLoader, pairNo, objNo, debug=False):
        self.cam = Camera()
        self.objNo = objNo
        self.pairNo = pairNo
        self.DL = dataLoader
        self.viewCount = 0
        self.cameraPos = {'r': 4, 'phi': pi / 2, 'theta': 0}  # position of camera in spherical coordinates
        self.defaultCam = {'r': 4, 'phi': pi / 2, 'theta': 0}
        self.moveSpeed = pi / 8
        self.offset = [0, 0, 0]
        self.debugMode = debug
        self.enhancedView = False
        self.poly = Polyhedron()  # Robots working model of what its looking at
        self.resolution = (1920, 1080)

    # gets the view based on where the camera currently is
    def processCameraView(self, debug=False):

        path = "views/robot" + str(self.objNo) + "_" + str(self.pairNo) + "_view" + str(self.viewCount) + ".png"
        viewExists = os.path.exists(path)
        print("View Exists: " + str(viewExists))
        if (not debug) or (not viewExists):

            print("Acquiring new view from web-api...")
            self.cam.view = self.DL.getView(self.pairNo, self.objNo, self.cam.getParams())
            fileName = os.getcwd() + "/views/robot" + str(self.objNo) + "_" + str(self.pairNo) + "_view" + str(
                self.viewCount) + ".png"
            cv2.imwrite(fileName, self.cam.view)

            if self.debugMode:
                print("Saved view: " + fileName)
        else:
            # If debug, load what would have been the processed view
            self.loadImg(path)

        self.enhancedView = False
        self.viewCount += 1

    def displayView(self):
        cv2.imshow("view", self.cam.view)
        cv2.waitKey(0)

    def showImg(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def loadImg(self, file):
        print("loading " + file)
        self.cam.view = cv2.imread(file, 0)
        self.enhancedView = False

    def point2world(self, point):
        #T = np.array([self.cam_x, self.cam_y, self.cam_z])  # translation
        #point = np.array(point)
        #R = [self.side, self.up, self.forward]
        ##print(self.rotMat)

        #return np.matmul(np.linalg.inv(R), (point))
        a = np.array(point)
        a = a - np.array([0, 0.1, 0])
        beta = -self.cameraPos['theta']
        #a = cos(beta)*a + sin(beta)*(np.cross(K, a)) + np.dot(K, a)*(1 - cos(beta))*K #invert theta rotation

        ROT = rotMatrix(0, 0, -beta)
        worldPoint = np.matmul(ROT, a)

        return worldPoint

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
            cv2.imwrite(
                "views/Centering_" + str(self.pairNo) + "_" + str(self.objNo) + "_" + str(self.viewCount - 1) + ".jpg",
                self.cam.view)
            return (cX, cY)

    # Calculates the 3D position based on a binocular shift b
    def calculate_pos(self, C1, C2, b,  reso=(1920, 1080), debug=False):
        # stereo-distance:

        f = 2100  # focus(mm) / sensor size(mm) * resolution width (px)
        d = np.abs(C1[0] - C2[0]) #disparity
        Z = (b * f) / d  # depth

        dp = (reso[0] // 2 - C1[0])  # distance from center when camera is at (0,0,R) in pixels for 1920x1080 image
        dy = Z * dp / f  # actual distance

        dp = (reso[1] // 2 - C1[1])
        dx = Z * dp / f
        dz = self.cameraPos["r"] - Z  # camera z-pos minus depth

        print("  >calculated distance to camera is: " + str(round(Z, 4)))
        print("  >Point is at " + str((dx, dy, dz)))

        if debug:
            print("<< calculate_pos debug START>>")
            print("C1", C1)
            print("C2", C2)
            #print("d", d)
            #print("b", b)
            #print("f", f)
            #print("b*f", b * f)
            print("Z", Z)
            print("dz", dz)
            #print("c1x, c2x", [C1[0], C2[0]])
            #print("|c1x - c2x|", np.abs(C1[0] - C2[0]))
            #print("Reso", reso)
            print("<< calculate_pos debug END>>")

        return np.round([dx, dy, dz], 8), round(Z, 8)

    # centers object in world coordinates
    def centerObject(self, debug=False):
        print("Stereoscopically centering object...")
        print("  >Getting first view")

        b = 0.2
        if not debug:
            self.setCam(4, 0, 0)
            self.processCameraView()
        else:
            self.cam.view = cv2.imread("views/robot" + str(self.objNo) + "_"+str(self.pairNo)+"_view0.png", 0)  # for debug
            self.setCam(4, 0, 0)
            self.viewCount = 1

        # get distance
        C1 = self.getCentroid()
        self.offset[1] = b

        print("  >Getting second view from different position (offset 0.2)")
        if not debug:
            self.setCam(4, 0, 0)
            self.processCameraView()
        else:
            self.cam.view = cv2.imread("views/robot" + str(self.objNo) + "_"+str(self.pairNo)+"_view1.png", 0)  # for debug
            self.setCam(4, 0, 0)
            self.viewCount = 2

        C2 = self.getCentroid()

        self.offset[0:3], zdist = self.calculate_pos(C1, C2, b, debug=False)

        if not debug:
            self.setCam(4, 0, 0)
            self.processCameraView()
        else:
            self.cam.view = cv2.imread("views/robot" + str(self.objNo) + "_"+str(self.pairNo)+"_view2.png", 0)  # for debug
            self.setCam(4, 0, 0)
            self.viewCount = 3

        CF = self.getCentroid()

        print("Object Centered")

        self.cameraPos = self.defaultCam

    # Contrasts colours in current view to better determine features. Achieves this by squaring and then filtering
    def enhanceView(self):
        if not self.enhancedView:
            # squaring to maxize face contrast
            print("Enhancing View Image")
            self.getROI()
            img = self.cam.view.copy()
            C= 25
            f = (131 * (C + 127)) / (127 * (131 - C))
            g = 127 * (1 - f)
            img = cv2.addWeighted(img, f, img, 0, g)
            ret, img = cv2.threshold(img, 127, 255 , 0)
            print(" > maximizing differences")
            self.cam.view = self.cam.view ** 2
            indices = np.where(img != 0)
            #print(indices)
            for x in range(0, len(indices[0])):  # Raise lower values more than high values
                ind = (indices[0][x], indices[1][x])
                # print(ind)
                B = self.cam.view[ind].copy()
                alpha = (255 - B) / 255
                self.cam.view[ind] += (int)(B * alpha)
            self.cam.view = self.cam.view ** 2

            for x in range(0, len(indices[0])):  # Raise lower values more than high values
                ind = (indices[0][x], indices[1][x])
                # print(ind)
                B = self.cam.view[ind].copy()
                alpha = (255 - B) / 255
                self.cam.view[ind] += (int)(100 * alpha)

            print(" > Normalizing")
            self.cam.vew = self.cam.view // np.max(self.cam.view)
            #showImg(self.cam.view)
            print(" > De-noising")

            self.cam.view = cv2.medianBlur(self.cam.view, 3)
            self.enhancedView = True
            cv2.imwrite("views/robot" + str(self.objNo) + "_" + str(self.pairNo) + "_" + "enhanced"+str(self.viewCount)+".png", self.cam.view)
            print("View Enhanced")
        else:
            print("View already enhanced")

        # return edges

    # <output>: vertices, faces, img
    # Finds features within current view. 'img' contains visual depiction of detected vertices
    def findFeatures(self):
        # ret, thresh = cv2.threshold(self.cam.view, 127, 255, 0)
        # self.cam.view = cv2.Canny(self.cam.view, 50, 255)
        # self.cam.view = cv2.dilate(self.cam.view, kernel=np.ones((3,3), np.float32))
        # self.cam.view = cv2.medianBlur(self.cam.view, 7)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # self.cam.view = cv2.drawContours(self.cam.view, contours, -1, (0, 255, 0), 3)

        if not self.enhancedView:
            print("(View needs to be enhanced)")
            self.enhanceView()
        img = cv2.cvtColor(self.cam.view, cv2.COLOR_GRAY2RGB)

        faces = self.findFaces()
        vertices = []
        removeFaces = []
        # find vertices on the face level
        for i in range(0, len(faces)):
            print("Determining vertices in face " + str(i))
            face = faces[i]
            tmp = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            dilations = 0
            numClusters = 0
            while numClusters < 3 and dilations < 5: # Can't have a face with < 3 points
                dilations += 1
                verts = self.detectCorners(face, dilations)
                # now process results from harris, find most likely vertices
                #clusters = self.getClusters(verts=verts)
                #newVerts = self.calcVertices(clusters)
                newVerts = self.removeColinearVertices(verts, face)
                numClusters = len(newVerts)
                print("Clusters in face " + str(numClusters))

            if dilations >= 5:
                print("Cannot make use of face, dilated too many times")
                removeFaces.append(i)
                continue

            for V in newVerts:
                cv2.circle(tmp, (V[1], V[0]), 5, (255 - i*50, 255 - i*50, i*50), 2)
            #showImg(tmp)
            vertices.extend(newVerts)
        newFaces = []
        for i in range(0, len(faces)):
            if i not in removeFaces:
                newFaces.append(faces[i])
        faces = newFaces

        # now do vertices globally to reduce redundancy
        clusters = self.getClusters_global(vertices)
        vertices = self.calcVertices(clusters)

        for V in vertices:
            cv2.circle(img, (V[1], V[0]), 5, (0, 0, 255), -1)
        cv2.imwrite("views/" + self.toString() + "vertices" + str(self.viewCount) + ".png", img)
        #showImg(img)
        return vertices, faces, img

    # <input> : face, an image mask depicting a face of the shape
    # <output>: corners, a list of detected corners
    def detectCorners(self, face, dilations):
        print(" > Detecting corners")
        img = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        #showImg(face)
        k = np.ones((5,5))
        face = cv2.erode(face, k, dilations)
        #showImg(face)
        face = cv2.dilate(face, k, dilations + 1)
        contours, hierarchy = cv2.findContours(face, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        #showImg(img)

        corners = []
        N = 1
        beta = 0.05
        realArea = cv2.countNonZero(face)
        cntArea = 0
        maxLoops = 5
        loops = 1
        while (not (realArea*0.90 <= cntArea <= realArea*1.05)) and (loops <= maxLoops):
            loops += 1
            print("    > ("+str(loops-1)+")Collecting corners in contour... b=" + str(beta))
            corners = []
            C = contours[0]
            cv2.drawContours(img, [C], -1, ((N-1)*50, 0, 255//N), 3)
            N += 1
            peri = cv2.arcLength(C, True)
            approx = cv2.approxPolyDP(C, beta*peri, True)
            for p in approx:
                pt = p[0]
                corners.append([pt[1], pt[0]])

            approxContour = np.array(approx)
            cntArea = cv2.contourArea(approxContour)

            beta -= 0.01
            print("          cntArea:          " + str(cntArea))
            print("         realArea:          " + str(realArea))
            print("         Potential Corners: "+str(len(approx)))
            print("        Is poly good appr?: "+str((realArea*0.9 <= cntArea <= realArea*1.05)))
        #showImg(img)
        return corners


    # <input> : verts, img array where white (255) points are potential vertices
    # <output>: clusters = [C1, C2, ...] where Ci =[p, p2, ...] are unique sets of points, that are close within maxDist
    def getClusters(self, verts=[], maxDist=15, pts=[]):
        # maxDist: maximum number of pixels away for something to be considered part of the cluster

        #tempImg = cv2.cvtColor(np.zeros_like(self.cam.view), cv2.COLOR_GRAY2RGB)

        # put get all high probability points
        if len(pts) == 0:
            for x in range(0, len(verts)):
                for y in range(0, len(verts[0])):
                    if verts[x, y] == 255:
                        pts.append([x, y])

        clusters = []  # init
        print(" > Clustering " + str(len(pts)) + " points")
        while pts:
            pt = pts.pop(0)

            # check clusters first, see if it belongs in any cluster
            next = False
            for C in clusters:
                for cp in C:
                    if dist_2d(pt, cp) <= maxDist:
                        C.append(pt)
                        next = True
                        break;
                if next:
                    break
            # if we added to  cluster, dont do anything more with this pt
            if next:
                continue

            # If we made it this far, then there was no cluster match
            clusters.append([pt])  # pt starts its own cluster

        # print(" > pruning clusters with too few members...")
        # for i in range(0, len(clusters)):
        #    if len(clusters[i]) <= 5:
        #        clusters.pop(i)

        print(" > Found " + str(len(clusters)) + " clusters")

        return np.array(clusters)

    # <input> : verts, array of point coordinates [x,y] which depict vertices detected for all faces
    # <output>: clusters = [C1, C2, ...] where Ci are unique sets of points
    # This function aggregates vertices accross all faces to remove redundancy (i.e. some verts belong to >1 faces)
    def getClusters_global(self, verts):
        pts = verts
        maxDist = 30
        clusters = []  # init
        print("Globally aggregating vertices from all faces")
        while pts:
            pt = pts.pop(0)

            # check clusters first, see if it belongs in any cluster
            next = False
            for C in clusters:
                for cp in C:
                    if dist_2d(pt, cp) <= maxDist:
                        C.append(pt)
                        next = True
                        break;
                if next:
                    break
            # if we added to  cluster, dont do anything more with this pt
            if next:
                continue

            # If we made it this far, then there was no cluster match
            clusters.append([pt])  # pt starts its own cluster

        return np.array(clusters)

    def getClusters_3D(self, pts, maxDist=1):
        clusters = []  # init
        print("Calculating clusters 3D")
        while pts:
            pt = pts.pop(0)
            # check clusters first, see if it belongs in any cluster
            next = False
            for C in clusters:
                for cp in C:
                    if dist_3d(pt, cp) <= maxDist:
                        C.append(pt)
                        next = True
                        break;
                if next:
                    break
            # if we added to  cluster, dont do anything more with this pt
            if next:
                continue

            # If we made it this far, then there was no cluster match
            clusters.append([pt])  # pt starts its own cluster
        print(" > Found " + str(len(clusters)) + " clusters")
        return np.array(clusters)

    # <input> : clusters = [C1, ... CN] where Ci = [[x1,y1], [x2,y2], ...] are unique sets of points
    # <output>: vertices = [V1, ... VN] where Vi = [x,y] is a unique point
    # This function calculates the most likely centroid for each cluster of points using a WLSQ method
    def calcVertices(self, clusters):
        vertices = []
        for i in range(0, len(clusters)):
            C = clusters[i]
            print(
                "   > Calculating weighted least squares centroid for cluster " + str(i + 1) + "/" + str(len(clusters)))
            wlsq = WLSQ(C)
            vertices.append(wlsq)

        return vertices

    # Finds faces in the current robot view. Working assumption is that face colour is unique and little variation
    # <output>: faces = [face1, face2, ...], where facei is an image mask within the current view depicting 1 face
    def findFaces(self):
        print("Localizing faces in current view")
        if not self.enhancedView:
            print("(View needs to be enhanced)")
            self.enhanceView()
        #showImg(self.cam.view)
        # Get histogram of colours
        print(" > Collecting colour histogram")
        numBuckets = 10
        bucketSize = 26
        colors, bins = np.histogram(self.cam.view.ravel(), 15, [0, 256])#cv2.calcHist([self.cam.view], [0], None, [numBuckets], [0, 256])
        print(colors)
        colors[0] = 0  # we don't care about black background
        faceColors = []
        last = 0
        for i in range(0, 15):
            C = colors[i]
            # if there are more than 1000 occurrences in that bucket, its probably a face
            if C >= 1500: # CHANGE BACK TO 1000 AFTER DONE DEBUGGING
                if last == 0:
                    last = int(bins[i])
                faceColors.append([bins[i], bins[i+1]])
                last = bins[i+1]

        print(" > Found " + str(len(faceColors)) + " faces")
        # now lets identify faces
        print(" > Generating masks")
        faces = []
        for fc in faceColors:
            #print(fc)
            face = np.zeros_like(self.cam.view)
            face = np.where(np.logical_and(self.cam.view < fc[1], self.cam.view >= fc[0]), 255, 0)
            face = face.astype(np.uint8)
            face = cv2.medianBlur(face, 5)
            faces.append(face)
            #showImg(face)


        print("Faces localized, masks generated")
        return faces

    # <input> : verts = [V1, V2, ...], Vi=[x,y], are predicted verts within 1 face
    # <output>: verts = [V1, V2, ...] is a subset of original verts
    # This function looks to see if any vertice lies directly between two others. This cannot occur in a real polygon,
    # thus they are removed
    def removeColinearVertices(self, verts, face):
        print("Removing collinear vertices")
        i = 0
        originalVerts = len(verts)

        while True:
            next = False
            for i in range(0, len(verts)):
                for j in range(0, len(verts)):
                    if j == i:
                        continue  # don't compare with itself
                    line = np.zeros_like(face)
                    V1 = verts[i]
                    V2 = verts[j]
                    cv2.line(line, (V1[1], V1[0]), (V2[1], V2[0]), 255, 5)  # draw a line from V1 to V2
                    #showImg(line)
                    for k in range(0, len(verts)):
                        if k == i or k == j:
                            continue  # not V1 or V2
                        V3 = verts[k]
                        if line[V3[0], V3[1]] == 255:
                            # V3 is in the line segment, therefore it is collinear
                            line[V3[0]-3:V3[0]+3, V3[1]-3:V3[1]+3] = 150
                            #self.showImg(line)
                            verts.pop(k)
                            next = True
                            break
                    if next:
                        break
                if next:
                    break

            if next:
                continue
            break # Will only leave the loop if it runs through all points and finds no collinear

        print("Removed (" + str(originalVerts - len(verts)) + ") vertices")
        return verts

    # <output>: pts = [p1, ... pn] where pi = (x,y,z) in real coordinates.
    #           verts1, list of original vertice locations in first view (pixel coordinates (x,y))
    #           face1, original face image masks
    def calcViewFeatures(self, dbgTkn=0, debug=False):
        # we want to move the camera some horizontal offset. The math here is that for small theta, sin(theta)=theta
        # The distance moved will be R*theta then, and if the movement is small enough we can approximate this as
        # purely horizontal
        print("Calculating view features (3D vertex positions and faces)")

        radius = self.cameraPos['r']

        self.processCameraView(debug=debug)
        verts1, faces1, keypoints1 = self.findFeatures()
        oldView = np.copy(self.cam.view)
        #self.showImg(keypoints1)

        b = 0.05 * (radius/2)
        self.shift(self.cam.side, -b)
        self.processCameraView(debug=debug)
        verts2, faces2, keypoints2 = self.findFeatures()
        newView = self.cam.view
        #self.showImg(keypoints2)

        self.shift(self.cam.side, b)

        #p0 = fixArrayForFlow(verts1)
        # We can't use verts2 because we have no idea which verts correspond to eachother in the new view.
        # To solve this, we use optical flow and figure out the corresponding points
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        #nextPts, st, err = cv2.calcOpticalFlowPyrLK(oldView, newView, p0, None, **lk_params)
        # Now lets see what corresponds to what
        #nextPts = unfuckFlowArray(nextPts)
        #newVert = []  # list of the 2nd verts with order corresponding to original verts


        # might not need optical flow, just using seeking
        # Our second view may not have the same number of corresponding verts if it didnt detect enough corners
        # This just ensures that we are only pairing points for which there is a correspondence.
        tmp = keypoints1.copy
        vertPairs = [] #pairs of corresponding vertices
        for i in range(0, len(verts1)):
            tmp = keypoints1.copy()
            v1 = verts1[i]
            new_i = self.seekRight(v1, verts2)
            cv2.circle(tmp, (v1[1], v1[0]), 5, (255,0,255), -1)
            if new_i != -1: #seek right found a valid point
                v2 = verts2.pop(new_i)
                vertPairs.append([v1, v2])
                cv2.circle(tmp, (v2[1], v2[0]), 5, (255,255,0), -1)
            else:
                print(" > Could not find corrensponding point for " + str(v1) + " in 2nd view")
            #showImg(tmp)

        vertPairs = np.array(vertPairs)

        # calculate the 3D positions for each point
        print("Calculating 3D poistions for detected points")
        pts = []
        print("SAME NUMBER OF FACES???? :" + str(len(faces1) == len(faces2)))


        #we probably picked up a bit of another face, so just get rid of the smallest face
        if not len(faces1) == len(faces2):
            print("  Adjusting faces")
            differences = []
            while len(faces1) != len(faces2):
                if len(faces2) > len(faces1):
                    for i in range(0, len(faces1)):
                        differences.append([])
                        area_i = cv2.countNonZero(faces1[i])
                        for j in range(0, len(faces2)):
                            area_j = cv2.countNonZero(faces2[j])
                            differences[i].append((np.abs(area_i - area_j)))

                    good_j = []
                    for i in range(0, len(faces1)):
                        best_j = 0
                        best = differences[i][0]
                        for j in range(1, len(faces2)):
                            if differences[i][j] < best:
                                best_j = j
                                best = differences[i][j]
                        good_j.append(best_j)

                    inters = list(set(good_j) & set(range(0, len(faces2))))
                    for j in range(0, len(faces2)):
                        if j not in inters:
                            faces2.pop(j)

                if len(faces2) < len(faces1):
                    for i in range(0, len(faces2)):
                        differences.append([])
                        area_i = cv2.countNonZero(faces2[i])
                        for j in range(0, len(faces1)):
                            area_j = cv2.countNonZero(faces1[j])
                            differences[i].append((np.abs(area_i - area_j)))

                    good_j = []
                    for i in range(0, len(faces2)):
                        best_j = 0
                        best = differences[i][0]
                        for j in range(1, len(faces1)):
                            if differences[i][j] < best:
                                best_j = j
                                best = differences[i][j]
                        good_j.append(best_j)

                    inters = list(set(good_j) & set(range(0, len(faces1))))
                    for j in range(0, len(faces1)):
                        if j not in inters:
                            faces1.pop(j)




        faceCorrelatedVerts = []
        for i in range(0, len(faces1)):
            p = self.correlateVertsWithFace(vertPairs[:, 0], faces1[i])
            faceCorrelatedVerts.append(p)

        faceCorrelatedVerts_2 = []
        for i in range(0, len(faces2)):
            p = self.correlateVertsWithFace(vertPairs[:, 1], faces2[i])
            faceCorrelatedVerts_2.append(p)

        faceCorrelatedPoints3D = []
        print(faceCorrelatedVerts)
        print(faceCorrelatedVerts_2)



        #ensure they are same length
        for i in range(0, len(faces1)):
            while len(faceCorrelatedVerts[i]) > len(faceCorrelatedVerts_2[i]):
                faceCorrelatedVerts[i].pop(-1)
            while len(faceCorrelatedVerts[i]) < len(faceCorrelatedVerts_2[i]):
                faceCorrelatedVerts_2[i].pop(-1)

        for j in range(0, len(faces1)):
            faceCorrelatedPoints3D.append([])
            for i in range(0, len(faceCorrelatedVerts[j])):
                V1 = faceCorrelatedVerts[j][i]
                V2 = faceCorrelatedVerts_2[j][i]
                temp = np.copy(keypoints1)
                cv2.circle(temp, (V1[1], V1[0]), 3, (255, 0, 255), 3)
                cv2.circle(temp, (V2[1], V2[0]), 3, (0, 255, 0), 3)
                cv2.imwrite("views/" + self.toString() + "_view" + str(self.viewCount) + "_corr" + str(i) + ".png",
                            img=temp)
                # invert the arrays
                p_cam, zdist = self.calculate_pos(V1[::-1], V2[::-1], b, self.resolution,
                                                  debug=False)  # in camera coordinates
                # ignore point if too far away to be accurate
                p_cam = np.round(p_cam, 8)

                faceCorrelatedPoints3D[j].append(self.point2world(p_cam))



        for i in range(0, len(vertPairs)):
            V1 = vertPairs[i][0]
            V2 = vertPairs[i][1]
            temp = np.copy(keypoints1)
            cv2.circle(temp, (V1[1], V1[0]), 3,  (255, 0, 255), 3)
            cv2.circle(temp, (V2[1], V2[0]), 3,  (0, 255, 0), 3)
            cv2.imwrite("views/" + self.toString() + "_view" + str(self.viewCount) + "_corr" + str(i) + ".png", img=temp)
            #invert the arrays
            p_cam, zdist = self.calculate_pos(V1[::-1], V2[::-1], b, reso=(1920 - 600, 1080-400), debug=False)  # in camera coordinates
            # ignore point if too far away to be accurate

            #if (zdist < self.cameraPos["r"] + 0.5):
            p_world = self.point2world(p_cam)
            p_cam = np.round(p_cam, 8)
            p_world = np.round(p_world, 8)
            pts.append(p_cam)



        return pts, vertPairs, faces1, faceCorrelatedPoints3D

    # Seeks the closest point to the right of p_left. If it doesn't find anything or its too far then index
    # Has no match and can't be used, returns -1
    def seekRight(self, p_left, points):
        maxSeek = 200 #will only look 200 pixels right
        dy = 25
        closestX = 99999999
        foundIndex = -1
        for i in range(0, len(points)):
            start = p_left
            p = points[i]
            AB = np.array(p) - np.array(start)
            if AB[1] <= maxSeek and np.abs(AB[0]) <= dy and p[1] > start[1]:
                if AB[1] < closestX:
                    closestX = AB[1]
                    foundIndex = i
        return foundIndex

    def getOptimalRadius(self):
        print("Determining Optimal Radius...")
        img = self.cam.view
        ret, img = cv2.threshold(img, 127, 255, 0)
        area = cv2.countNonZero(img)
        totalArea = self.resolution[0]*self.resolution[1]
        coverage = round(float(area/totalArea), 8)
        print("coverage:", coverage)
        if coverage < 0.01:
            return 1.5
        elif coverage <0.015:
            return 2.5
        elif coverage < 0.05:
            return 3
        else:
            return 4

    # This function calculated the 3D positions of vertices from the current orientation at 5 different radii
    # It then aggregates these values and calculates the best guess
    def estimateTrueViewFeatures(self, debug=False):
        pts = []
        self.processCameraView(debug=debug)
        self.set_radius(self.getOptimalRadius())
        ps, vertPairs, FACES, FCP = self.calcViewFeatures('2.6', debug=debug)
        pts.extend(ps)


        color = np.array([1,1,1])


        self.poly.addFace(FCP)

    # <input> : verts = [v1, ...] vi = (x,y) in pixel coordinates
    #           face, 2D image mask of detected face
    # <output>: vertsInFace, a subset of verts which belong to the face. If verts on the face is less than 3, returns -1
    def correlateVertsWithFace(self, verts, face):
        face = np.array(face)
        k=np.ones((3,3))

        verts = np.array(verts)
        vertsInFace = []
        loops = 1
        while len(vertsInFace) < 3:
            print(" > ("+str(loops)+")Correlating Vertices with Face")
            face = cv2.dilate(face, kernel=k, iterations=loops)  # Because im tired of it missing narrow corners
            faceRGB = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # gives us more channels to work with
            for v in verts:
                r = 8  # search radius
                subFace = face[v[0] - r:v[0] + r, v[1] - r:v[1] + r]
                isInFace = False
                for row in subFace:
                    for pix in row:
                        if pix == 255:
                            isInFace = True
                            break
                    if isInFace:
                        break
                if isInFace:
                    cv2.circle(faceRGB, (v[1], v[0]), radius=r, color=(0, 255, 0), thickness=r)
                    vertsInFace.append(v)
                else:
                    cv2.circle(faceRGB, (v[1], v[0]), radius=r, color=(0, 0, 255), thickness=r)
            #showImg(faceRGB)
            print("   verts in face: " + str(len(vertsInFace)))

        return vertsInFace

    # shrinks image, less computations needed
    def getROI(self):
        self.cam.view = self.cam.view[100:1080 - 100, 400: 1920 - 400]
        width = int(self.cam.view.shape[1] * 0.7)
        height = int(self.cam.view.shape[0] * 0.7)
        dim = (width, height)
        self.cam.view = cv2.resize(self.cam.view, dim, interpolation=cv2.INTER_AREA)
        self.resolution = (self.cam.view.shape[1], self.cam.view.shape[0])

    #re-converts ROI coordinate point to 1920/1080 point coord
    def ROI_to_full(self, point):
        point[0] = point[0] + 450
        point[1] = point[1] + 250
        return point

    # interface functions ==============================================================================================
    # moves camera in directions specified by fn name, by amt (radians)
    def move_down(self, amt=0):
        i = 1
        if amt != 0:
            i = 0
        newPhi = self.cameraPos["phi"] - self.moveSpeed * i + amt
        if 0 <= newPhi:
            self.cameraPos["phi"] = newPhi
            print("moved DOWN to " + str(self.cameraPos))

            if round(newPhi, 8) == 0:
                newPhi = 0.00001  # dont want it exactly at singularity
                self.cameraPos["phi"] = newPhi

        else:
            print("Cannot move UP, already at maximum")
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def move_up(self, amt=0):
        i = 1
        if amt != 0:
            i = 0
        newPhi = self.cameraPos["phi"] + self.moveSpeed * i + amt
        if newPhi <= pi:
            self.cameraPos["phi"] = newPhi
            print("moved UP to " + str(self.cameraPos))  # tell the folks at home

            if round(newPhi, 8) == pi:
                newPhi = pi - 0.00001  # dont want it exactly at singularity
                self.cameraPos["phi"] = newPhi
        else:
            print("Cannot move DOWN, already at minimum")
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def move_left(self, amt=0):
        i = 1
        if amt != 0:
            i = 0
        newTheta = self.cameraPos["theta"] - self.moveSpeed * i + amt
        self.cameraPos["theta"] = newTheta % (2 * pi)
        print("moved LEFT to " + str(self.cameraPos))
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def move_right(self, amt=0):
        i = 1
        if amt != 0:
            i = 0
        newTheta = self.cameraPos["theta"] + self.moveSpeed * i + amt
        self.cameraPos["theta"] = newTheta % (2 * pi)
        print("moved RIGHT to " + str(self.cameraPos))
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    def set_radius(self, r):
        self.cameraPos["r"] = r
        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])
        print("Radius set to " + str(r))

    # translates camera in dir by amt (units)
    def shift(self, dir, amt):
        print("Shifting camera by " + str(amt) + " in dir " + str(dir))
        offset = np.array(self.offset)
        offset = offset + np.array(dir) * amt

        self.offset[0] = offset[0]
        self.offset[1] = offset[1]
        self.offset[2] = offset[2]

        self.setCam(self.cameraPos["r"], self.cameraPos["phi"], self.cameraPos["theta"])

    #moves camera to a cartesian point in space
    def move_to(self, point3D):
        p = point3D
        C = 0

        r = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        t = np.arccos(p[2]/r)
        p = np.arctan(p[1]/p[0])

        self.setCam(r, p, t)

    def findIntersectWithSphere(self, lineStart, direction, radius):
        O = np.array(lineStart)
        u = np.array(direction)
        #quadratic formula
        a = 1
        b = 2*np.dot(u,O)
        c = np.dot(O,O) - radius**2

        dscrm = np.round(b**2 - 4*a*c, 8)
        d = 0

        if dscrm < 0:
            print("SPHERE INTERSECTION: NONE EXIST")
            return None
        if dscrm == 0: # one solution
            d = -b/2
        else: #two solutions
            d1 = 0.5*(-b + np.sqrt(b**2 -4*a*c))
            d2 = 0.5*(-b - np.sqrt(b**2 -4*a*c))
            d = np.max([d1, d2]) #we wanna go in the position u direction

        p = O + d*u
        return p
    # orients the camera in spherical coordinates around the camera focus point (default is 0)
    def setCam(self, r, phi, theta):
        # position the camera
        self.cam.cam_x = np.round(r * np.cos(theta) * np.sin(phi) + self.offset[0], 8)
        self.cam.cam_y = np.round(r * np.sin(theta) * np.sin(phi) + self.offset[1], 8)
        self.cam.cam_z = np.round(r * np.cos(phi) + self.offset[2], 8)

        if round(phi, 8) == 0:
            phi = 0.001 #signularity avoidance

        self.cameraPos["r"] = r
        self.cameraPos["theta"] = theta
        self.cameraPos["phi"] = phi

        # set orientation to look at target

        # spherical unit vectors
        phiHat = np.round([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)], 8)
        thetaHat = np.round([-np.sin(theta), np.cos(theta), 0], 8)
        rHat = np.round([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 8)

        # point to the center of the view sphere
        self.cam.lookAt(self.offset)

        if self.debugMode:
            print("Camera oriented to", self.cam.getParams())
            # save transcript of orientations and positions
            fileName = os.getcwd() + "/views/movement.txt"
            f = open(fileName, 'a')
            f.write(self.cam.printParams())
            f.close()

    # Util =============================================================================================================
    def toString(self):
        return "robot" + str(self.objNo) + "_" + str(self.pairNo)

#  Unit Tests ----------------------------------------------------------------------------------------------------------

def TestCenterObject(robot):
    print(">>>> ROBOT "+str(robot.objNo)+"."+str(robot.pairNo) +" centerObject")
    robot.centerObject(debug=True)

def TestVerticeCount(robot):
    print(">>>> Testing vertice counting")
    NUM_VERTS = 4
    robot.loadImg("views/robot0_1_view3.png")
    # robot.processCameraView()
    robot.enhanceView()
    robot.findVertices()

def TestFindFaces(robot):
    print(">>>> Testing Find Faces")
    robot.move_right(0.2)
    robot.move_up(0.8)
    robot.loadImg("views/robot1_1_view8.png")
    verts, faces, img = robot.findFeatures()
    robot.showImg(img)

def TestCamera2World(robot):

    tp = [1, 0, 0]

    robot.setCam(5, phi=pi/2, theta=0)
    print("forward", robot.cam.forward)
    print("up     ", robot.cam.up)
    print("right  ", robot.cam.side)
    p = robot.cam.world2cam(tp)
    print("Actual  ", np.round(p, 8))
    print("Expected", [0, 0, -1])

    robot.setCam(5, phi=pi/2, theta=pi/2)
    print("forward", robot.cam.forward)
    print("up     ", robot.cam.up)
    print("right  ", robot.cam.side)
    p = robot.cam.world2cam(tp)
    print("Actual  ", np.round(p, 8))
    print("Expected", [-1, 0, 0])

    robot.setCam(5, phi=pi/2, theta=pi)
    print("forward", robot.cam.forward)
    print("up     ", robot.cam.up)
    print("right  ", robot.cam.side)
    p = robot.cam.world2cam(tp)
    print("Actual  ", np.round(p, 8))
    print("Expected", [0, 0, 1])

    robot.setCam(5, phi=pi/2, theta=3*pi/2)
    print("forward", robot.cam.forward)
    print("up     ", robot.cam.up)
    print("right  ", robot.cam.side)
    p = robot.cam.world2cam(tp)
    print("Actual  ", np.round(p, 8))
    print("Expected", [1, 0, 0])

def TestPointDepth(robot):
    realPts = [[0.5,   0.5, 0],
               [0.5,  -0.5, 0],
               [-0.5,  0.5, 0],
               [-0.5, -0.5, 0]]
    print(robot.toString() + " >> Point Depth Debugging")
    robot.set_radius(3)
    pts1 = robot.calcViewFeatures('3')
    robot.set_radius(4)
    pts2 = robot.calcViewFeatures('4')
    robot.set_radius(5)
    pts3 = robot.calcViewFeatures('5')
    robot.set_radius(8)
    pts4 = robot.calcViewFeatures('8')

    print("r=3")
    print(np.mean(np.array(pts1).transpose()[2]))
    print(pts1)
    print("-----")
    print("r=4")
    print(np.mean(np.array(pts1).transpose()[2]))
    print(pts2)
    print("-----")
    print("r=5")
    print(np.mean(np.array(pts3).transpose()[2]))
    print(pts3)
    print("-----")
    print("r=8")
    print(np.mean(np.array(pts4).transpose()[2]))
    print(pts4)

def TuneFocus(robot):
    robot.set_radius(4)


def TestCongruency():
    RESULTS = []
    for i in range(30, 48, 2):
        img1 = cv2.imread("views/robot0_1_view" + str(i) + ".png", 0)
        img2 = cv2.imread("views/robot1_1_view" + str(i) + ".png", 0)
        ret, img1 = cv2.threshold(img1, 127, 255, 0)
        ret, img2 = cv2.threshold(img2, 127, 255, 0)

        if cv2.countNonZero(img1) == 0:
            print("Dont see anything for face 1")
            continue
        if cv2.countNonZero(img2) == 0:
            print("Dont see anything for face 2")
            continue

        res = compareImages(img1, img2)
        res.append(['Views', i, i])
        RESULTS.append(res)
    writeToCSV(1, RESULTS)

    RESULTS = []
    for i in range(18, 27):
        img1 = cv2.imread("views/robot0_2_view" + str(i) + ".png", 0)
        img2 = cv2.imread("views/robot1_2_view" + str(i) + ".png", 0)
        ret, img1 = cv2.threshold(img1, 127, 255, 0)
        ret, img2 = cv2.threshold(img2, 127, 255, 0)

        res = compareImages(img1, img2)
        res.append(['Views', i, i])
        RESULTS.append(res)
    writeToCSV(2, RESULTS)

    RESULTS = []
    for i in range(18, 28):
        img1 = cv2.imread("views/robot0_3_view" + str(i) + ".png", 0)
        img2 = cv2.imread("views/robot1_3_view" + str(i) + ".png", 0)
        ret, img1 = cv2.threshold(img1, 127, 255, 0)
        ret, img2 = cv2.threshold(img2, 127, 255, 0)

        res = compareImages(img1, img2)
        res.append(['Views', i, i])
        RESULTS.append(res)
    writeToCSV(3, RESULTS)

    RESULTS = []
    for i in range(18, 21):
        img1 = cv2.imread("views/robot0_4_view" + str(i) + ".png", 0)
        img2 = cv2.imread("views/robot1_4_view" + str(i) + ".png", 0)
        ret, img1 = cv2.threshold(img1, 127, 255, 0)
        ret, img2 = cv2.threshold(img2, 127, 255, 0)

        res = compareImages(img1, img2)
        res.append(['Views', i, i])
        RESULTS.append(res)
    writeToCSV(4, RESULTS)

    RESULTS = []
    for i in range(18, 28):
        img1 = cv2.imread("views/robot0_5_view" + str(i) + ".png", 0)
        img2 = cv2.imread("views/robot1_5_view" + str(i) + ".png", 0)
        ret, img1 = cv2.threshold(img1, 127, 255, 0)
        ret, img2 = cv2.threshold(img2, 127, 255, 0)

        res = compareImages(img1, img2)
        res.append(['Views', i, i])
        RESULTS.append(res)
    writeToCSV(5, RESULTS)

    RESULTS = []
    for i in range(18, 21):
        img1 = cv2.imread("views/robot0_6_view" + str(i) + ".png", 0)
        img2 = cv2.imread("views/robot1_6_view" + str(i) + ".png", 0)
        ret, img1 = cv2.threshold(img1, 127, 255, 0)
        ret, img2 = cv2.threshold(img2, 127, 255, 0)

        res = compareImages(img1, img2)
        res.append(['Views', i, i])
        RESULTS.append(res)
    writeToCSV(6, RESULTS)



# Actual Algorithm ---------------------------------------------

def TestFindFeatures(robot, dbg=False, init=0):
    print(">>>> Testing feature correspondance " + robot.toString())


    robot.move_right(amt=init) #Random starting point so we can retry if theres a fucky wucky

    robot.estimateTrueViewFeatures(debug=dbg)
    robot.move_right(amt=pi/3)
    robot.estimateTrueViewFeatures(debug=dbg)
    robot.move_right(amt=pi/3)
    robot.estimateTrueViewFeatures(debug=dbg)
    robot.move_right(amt=pi/3)
    robot.estimateTrueViewFeatures(debug=dbg)
    robot.move_right(amt=pi/3)
    robot.estimateTrueViewFeatures(debug=dbg)
    robot.move_right(amt=pi/3)
    robot.estimateTrueViewFeatures(debug=dbg)

    #robot.poly.viewPoints()
    robot.poly.calcBoundingVolume()
    print("Faces: " + str(robot.poly.numFaces()))
    robot.poly.pruneFaces()
    print("Faces: " + str(robot.poly.numFaces()))
    #robot.poly.drawFaces()

    return robot.poly.numFaces()


def comparePolygons(robot1, robot2, dbg=False):
    poly1 = robot1.poly
    poly2 = robot2.poly

    comparison = poly1.comparePoly(poly2)

    RESULTS = []
    #lets try to find similar views

    print("Looking for possible same views in faces1(" + str(len(poly1.faces)) +"), faces2("+str(len(poly2.faces))+")")

    for face1 in poly1.faces:
        for face2 in poly2.faces:
            if face2.area * 0.9 <= face1.area <= face2.area * 1.1:  # if the areas are pretty close
                print("I found two possible similar faces")
                norm1 = face1.normal
                norm2 = face2.normal
                COM1 = face1.COM
                COM2 = face2.COM

                X = robot1.findIntersectWithSphere(COM1, norm1, 4)
                if X is None:
                    break
                robot1.move_to(X)
                X = robot2.findIntersectWithSphere(COM2, norm2, 4)
                if X is None:
                    continue
                robot2.move_to(X)

                #robot1.processCameraView(debug=dbg)
                r = robot1.getOptimalRadius()
                robot1.set_radius(r)
                print("Getting a look at face 1")
                robot1.processCameraView(debug=dbg)
                robot1.getROI()

                #robot2.processCameraView(debug=dbg)
                r = robot2.getOptimalRadius()
                robot2.set_radius(r)
                print("Getting a look at face 2")
                robot2.processCameraView(debug=dbg)
                robot2.getROI()

                ret, img1 = cv2.threshold(robot1.cam.view, 127, 255, 0)
                ret, img2 = cv2.threshold(robot2.cam.view, 127, 255, 0)

                if cv2.countNonZero(img1) == 0:
                    print("Dont see anything for face 1")
                    break
                if cv2.countNonZero(img2) == 0:
                    print("Dont see anything for face 2")
                    continue

                result = compareImages(img1, img2)
                result.append(['Views', robot1.viewCount, robot2.viewCount])
                result.append(['BoundingVolumes', robot1.poly.boundingVolume, robot2.poly.boundingVolume])
                RESULTS.append(result)
                print(result)
            else:
                print("Not right!")
    #print(RESULTS)
    return RESULTS

def compareImages(img1, img2):
    IMG = np.concatenate((img1, img2), axis=1)
    #showImg(IMG)

    width = int(img1.shape[1] * 0.3)
    height = int(img1.shape[0] * 0.3)
    dim = (width, height)

    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    views = [img1, img2]
    polygons = []
    perims = []

    #Define the support polygons for each face view
    for i in range(0,2):
        #print(" > Defining best polygon for " + str(i))
        face = views[i]
        img = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        contours, hierarchy = cv2.findContours(face, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        corners = []
        N = 1
        beta = 0.05
        realArea = cv2.countNonZero(face)
        cntArea = 0
        maxLoops = 5
        loops = 1
        while (not (realArea * 0.95 <= cntArea <= realArea * 1.05)) and (loops <= maxLoops):
            loops += 1
            #print("    > (" + str(loops) + ")Collecting corners in contour... b=" + str(beta))
            corners = []
            cv2.drawContours(img, contours[0], -1, ((N - 1) * 50, 0, 255 // N), 3)
            N += 1
            peri = cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], beta * peri, True)
            for p in approx:
                pt = p[0]
                corners.append([pt[1], pt[0]])
                cv2.circle(img, (pt[0], pt[1]), 3, (255,255,0), -1)
                approxContour = np.array(approx)
                cntArea = cv2.contourArea(approxContour)
            # showImg(img)
            beta -= 0.01
        #showImg(img)
        polygons.append(corners)
        perims.append(peri)


    COMS = []
    for i in range(0,2):
        COM = np.array([0, 0])
        for p in polygons[i]:
            COM += np.array(p)
        COM = COM / len(polygons[i])
        COMS.append(COM)

    #move points all to COM origin at (0,0). Here we're kinda breaking away from pixel representation
    #print(polygons)

    for i in range(0,2):
        for j in range(0, len(polygons[i])):
            polygons[i][j] -= COMS[i]

        # lets find the largest distance so we can normalize:

    poly1, best1 = getCongruency(polygons.copy(), COMS, img1, 0) #Different starting points
    poly2, best2 = getCongruency(polygons.copy(), COMS, img1, pi/2)
    poly3, best3 = getCongruency(polygons.copy(), COMS, img1, pi)
    poly4, best4 = getCongruency(polygons.copy(), COMS, img1, 3*pi/2)
    #print([best1, best2, best3, best4])
    bests = np.round([best1, best2, best3, best4], 8)
    Best = 99999
    smol = np.min(np.round(bests, 8))
    if smol == bests[0]:
        polygons = poly1
        Best = best1
    elif smol == bests[1]:
        polygons = poly2
        Best = best2
    elif smol == bests[2]:
        polygons = poly3
        Best = best3
    elif smol == bests[3]:
        polygons = poly4
        Best = best4

    #print("Best Congruency: " + str(Best))

    for i in range(0,2):
        for j in range(0, len(polygons[i])):
            polygons[i][j] += COMS[0]
            polygons[i][j] = polygons[i][j].astype(int)
            polygons[i][j] = polygons[i][j][::-1]
    face = np.zeros_like(img1)
    img = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    for i in range(0,2):
        for p in polygons[i]:
            cv2.circle(img, (int(COMS[0][1]), int(COMS[0][0])), 4, (0, 255, 0), 4)

    face = np.zeros_like(img1)
    imgA = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    imgB = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    cv2.fillPoly(imgA, [np.array(polygons[0]).astype(int)], (0, 0, 255))
    cv2.fillPoly(imgB, [np.array(polygons[1]).astype(int)], (255, 0, 0))
    img = imgA + imgB
    #showImg(img)
    mn = np.min(perims)
    mx = np.max(perims)

    return [['Perimiter Ratio', mn/mx], ['Best Congruency', Best]]

def getCongruency(polygons, COMS, img1, start):

    bestPoly = []

    maxDist = 0
    for x in range(0, len(polygons[0])):
        for y in range(0, len(polygons[0])):
            if x != y:
                p1 = polygons[0][x]
                p2 = polygons[0][y]
                d = dist_2d(p1, p2)
                if d > maxDist:
                    maxDist = d

    maxDist2 = 0
    for x in range(0, len(polygons[1])):
        for y in range(0, len(polygons[1])):
            if x != y:
                p1 = polygons[1][x]
                p2 = polygons[1][y]
                d = dist_2d(p1, p2)
                if d > maxDist2:
                    maxDist2 = d
    #for i in range(0, len(polygons[1])):
        #polygons[1][i] *= maxDist/maxDist2 #scale them similarily

    # print(polygons)
    # now both are centered at (0,0)
    congruency = 9999999
    Best = 999999
    T = start
    step = pi/6
    for i in range(0, 20):
        ROT = [[cos(T), -sin(T)],
               [sin(T), cos(T)]]
        for k in range(0, len(polygons[1])):
            b = np.array(polygons[1][k])
            polygons[1][k] = np.matmul(ROT, b)  # Rotate all points

        ##################### View Rotation in process #############################
        for i in range(0, 2):
            for j in range(0, len(polygons[i])):
                polygons[i][j] += COMS[0]
                polygons[i][j] = polygons[i][j][::-1]

        face = np.zeros_like(img1)
        imgA = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        imgB = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        cv2.fillPoly(imgA, [np.array(polygons[0]).astype(int)], (0, 0, 255))
        cv2.fillPoly(imgB, [np.array(polygons[1]).astype(int)], (255, 0, 0))
        img = imgA + imgB

        for i in range(0, 2):
            for j in range(0, len(polygons[i])):
                polygons[i][j] = polygons[i][j][::-1]
                polygons[i][j] -= COMS[0]
        # print("Congruency: " + str(congruency)+", step:" + str(step))
        #showImg(img)

        red = np.sum(np.where((img == (0,0,255)).all(axis=2)))
        blue = np.sum(np.where((img == (255,0,0)).all(axis=2)))
        purple = np.sum(np.where((img == (255,0,255)).all(axis=2)))
        areaRatio = (red + blue)/(red + blue + purple)


        ###########################################################################

        alpha = 0.8 #area penalty coefficient
        beta = 0.8
        # Calculate congruency
        pointDists = 0
        for p1 in polygons[0]:
            shortest = 99999
            for p2 in polygons[1]:
                if dist_2d(p1, p2) < shortest:
                    shortest = dist_2d(p1, p2)
            pointDists += shortest ** 2

        newCongruency = alpha*areaRatio + beta*pointDists/(maxDist ** 2)

        #print("pentalty", (areaRatio*alpha))
        #print("congruency", newCongruency)\


        #newCongruency /= 2
        if newCongruency - congruency > 0:  # i.e. we got worse
            step = -step*0.95
        else:
            step = 0.8*step
        T = step
        congruency = newCongruency
        if congruency < Best:
            Best = congruency
            bestPoly = polygons
    print("area", (alpha * areaRatio))
    print("point", beta * pointDists / (maxDist ** 2))
    return bestPoly, Best

def decideSameDiff(comparison):
    pass

def writeToCSV(pair, results):

    with open('Analysis/Results.csv', mode='a') as csv:
        for row in results:
            print("saving " + str(row))
            line = str(pair) + "," + str(row[2][1]) + '-' + str(row[2][2]) + "," + str(row[0][1]) + "," + str(row[1][1]) +','+str(row[3][1])+','+str(row[3][2])+ "\n"
            csv.write(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR - SameDiff.py requires 1 argument <csvpath>")
        exit(0)

    if os.path.exists("views/movement.txt"):
        os.remove("views/movement.txt")

    file = str(sys.argv[1])  # Can replace this with the path of the file
    DL = DataLoader(file)

    NumPairs = len(DL.objects)

    #TestCongruency()

    Results = []
    #views = ['19', '25', '23', '21', '29']
    #for v in views:
        #img1 = cv2.imread("views/robot0_3_view"+v+".png", 0)
        #img2 = cv2.imread("views/robot1_3_view28.png", 0)
        #print(compareImages(img1, img2))
    for i in range(1, 10):
        Robot1 = Robot(DL, i, 0, debug=False)
        Robot2 = Robot(DL, i, 1, debug=False)

        faces1 = TestFindFeatures(Robot1, True, init=1.0903376996637926)
        faces2 = TestFindFeatures(Robot2, True, init=1.1596437800060373)

        print("Same # Faces?: " + str(faces1==faces2))
        res = comparePolygons(Robot1, Robot2, True)
        writeToCSV(i, res)
        Results.append(res)



    #TestFindFaces(robot2)

    # DONE, software will ding when done processing
    print("======================================================")
    print("      //////   //    //   ////                        ")
    print("      //       ////  //   //  //                      ")
    print("      //////   // // //   //   //                     ")
    print("      //       //  ////   //  //                      ")
    print("      //////   //    //   ////                        ")
    print("======================================================")
    print("I HAVE FINISHED, THANK YOU HAVE A NICE DAY")
    playsound("Toaster Oven Ding - Sound Effect.mp3")

    for i in range(0, len(Results)):
        print("For pair " + str(i+1))
        for r in Results[i]:
            print(r)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    #robot2.poly.viewPoints()


