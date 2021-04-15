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
    return a / N

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
    def point2world(self, point):
        T = np.array([self.cam_x, self.cam_y, self.cam_z])  # translation
        point = np.array(point)
        R = [self.side, self.up, self.forward]
        #print(self.rotMat)

        return np.matmul(np.linalg.inv(R), (point))

    def world2cam(self, point):
        point = np.array(point)
        R = [self.side, self.up, self.forward]
        return np.matmul(R, point)

class Polyhedron:

    def __init__(self):
        self.points = []  # list of all points in 3D
        self.colors = []
        self.faces = []  # list of groups of points which make up faces
        self.pointCloud = open3d.geometry.PointCloud()

    def addFace(self, fcs):
        self.faces.extend(fcs)

    def addPoints(self, pts, color=[np.random.random(), np.random.random(), np.random.random()]):
        #color = [np.random.random(), np.random.random(), np.random.random()]
        self.points.extend(pts)
        for i in range(0, len(pts)):
            self.colors.append(color)

    def numFaces(self):
        return len(self.faces)

    def numPoints(self):
        return len(self.points)

    def viewPoints(self):
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

        if not debug:

            print("Acquiring new view from web-api...")
            self.cam.view = self.DL.getView(self.pairNo, self.objNo, self.cam.getParams())
            fileName = os.getcwd() + "/views/robot" + str(self.objNo) + "_" + str(self.pairNo) + "_view" + str(
                self.viewCount) + ".png"
            cv2.imwrite(fileName, self.cam.view)

            if self.debugMode:
                print("Saved view: " + fileName)

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
            print(" > maximizing differences")
            self.cam.view = self.cam.view ** 2
            print(" > Normalizing")
            self.cam.vew = self.cam.view // np.max(self.cam.view)
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
        # find vertices on the face level
        for i in range(0, len(faces)):
            print("Determining vertices in face " + str(i))
            face = faces[i]
            verts = self.detectCorners(face)
            # now process results from harris, find most likely vertices
            clusters = self.getClusters(verts=verts)
            newVerts = self.calcVertices(clusters)
            newVerts = self.removeColinearVertices(newVerts, face)
            vertices.extend(newVerts)
        # now do vertices globally to reduce redundancy
        clusters = self.getClusters_global(vertices)
        vertices = self.calcVertices(clusters)

        for V in vertices:
            cv2.circle(img, (V[1], V[0]), 5, (0, 0, 255), -1)
        cv2.imwrite("views/" + self.toString() + "vertices" + str(self.viewCount) + ".png", img)

        return vertices, faces, img

    # <input> : face, an image mask depicting a face of the shape
    # <output>: verts, a 2D image array where white dots (255) are corner points detected by harris corner detection
    def detectCorners(self, face):
        verts = np.zeros_like(face)

        img = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        # do harris corner detection at multiple settings, then aggregate results
        dst = cv2.cornerHarris(face, 3, 7, 0.015)
        dst = cv2.dilate(dst, None)
        verts[dst > 0.01 * dst.max()] = 50

        dst = cv2.cornerHarris(face, 2, 5, 0.02)
        dst = cv2.dilate(dst, None)
        verts[dst > 0.01 * dst.max()] += 50

        dst = cv2.cornerHarris(face, 2, 5, 0.025)
        dst = cv2.dilate(dst, None)
        verts[dst > 0.01 * dst.max()] += 50

        dst = cv2.cornerHarris(face, 2, 5, 0.03)
        dst = cv2.dilate(dst, None)
        verts[dst > 0.01 * dst.max()] += 50

        dst = cv2.cornerHarris(face, 2, 5, 0.04)
        dst = cv2.dilate(dst, None)
        verts[dst > 0.01 * dst.max()] += 50

        # self.showImg(verts)

        # dst = cv2.cornerHarris(self.cam.view, 2, 5, 0.05)
        # dst = cv2.dilate(dst, None)
        # 0verts[dst > 0.01 * dst.max()] += 50

        # dst = cv2.cornerHarris(self.cam.view, 2, 5, 0.07)
        # dst = cv2.dilate(dst, None)
        # verts[dst > 0.01 * dst.max()] += 50

        verts[verts >= 100] = 255
        verts[verts != 255] = 0

        print(" > Found " + str(len(verts)) + " potential corners")
        return verts

    # <input> : verts, img array where white (255) points are potential vertices
    # <output>: clusters = [C1, C2, ...] where Ci =[p, p2, ...] are unique sets of points, that are close within maxDist
    def getClusters(self, verts=[], maxDist=25, pts=[]):
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
        maxDist = 25
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

        # Get histogram of colours
        print(" > Collecting colour histogram")
        bucketSize = 26
        colors = cv2.calcHist([self.cam.view], [0], None, [10], [0, 256])

        colors[0] = 0  # we don't care about black background
        faceColors = []
        for i in range(0, 10):
            C = colors[i]
            # if there are more than 1000 occurrences in that bucket, its probably a face
            if C >= 1000: # CHANGE BACK TO 1000 AFTER DONE DEBUGGING
                faceColors.append([i * bucketSize, (i + 1) * bucketSize])

        print(" > Found " + str(len(faceColors)) + " faces")
        # now lets identify faces
        print(" > Generating masks")
        faces = []
        for fc in faceColors:
            face = np.zeros_like(self.cam.view)
            # loop through view
            for x in range(0, len(self.cam.view)):
                for y in range(0, len(self.cam.view[0])):
                    if fc[0] <= self.cam.view[x, y] < fc[1]:
                        face[x, y] = 255
            face = cv2.medianBlur(face, 5)
            faces.append(face)

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

    # calculates object features (vertices, faces) from current view
    def calcViewFeatures(self, dbgTkn=0):
        # we want to move the camera some horizontal offset. The math here is that for small theta, sin(theta)=theta
        # The distance moved will be R*theta then, and if the movement is small enough we can approximate this as
        # purely horizontal
        print("Calculating view features (3D vertex positions and faces)")

        radius = self.cameraPos['r']

        self.processCameraView()
        #self.loadImg("BlenderViews/square_r"+dbgTkn+".png")
        #self.enhancedView = True
        #self.showImg(self.cam.view)
        verts1, faces1, keypoints1 = self.findFeatures()
        oldView = np.copy(self.cam.view)
        #self.showImg(keypoints1)

        b = 0.05 * (radius/2)
        self.shift(self.cam.side, -b)
        self.processCameraView()
        #self.loadImg("BlenderViews/square_r"+dbgTkn+"_right.png")
        #self.enhancedView = True
        verts2, faces2, keypoints2 = self.findFeatures()
        newView = self.cam.view
        #self.showImg(keypoints2)

        self.shift(self.cam.side, b)

        p0 = fixArrayForFlow(verts1)
        # We can't use verts2 because we have no idea which verts correspond to eachother in the new view.
        # To solve this, we use optical flow and figure out the corresponding points
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        nextPts, st, err = cv2.calcOpticalFlowPyrLK(oldView, newView, p0, None, **lk_params)
        # Now lets see what corresponds to what
        nextPts = unfuckFlowArray(nextPts)
        newVert = []  # list of the 2nd verts with order corresponding to original verts

        # loop through vertices, find the closest one, then it must correspond and add to new array which now is ordered
        # We do this cause our original vertices are much more accurate, unless we didnt detect enough
        if len(verts2) != len(verts1):
            newVert = nextPts
            print("Incorrect number of verts between views, using optical flow values")
        else:
            banned = []
            for i in range(0, len(nextPts)):
                pp = nextPts[i][::-1]
                closest = 99999999
                index = -1
                for k in range(0, len(verts2)):
                    if k in banned:  # if we already looked here. Just to gaurentee no duplicates
                        continue
                    p = verts2[k]
                    #print("real " + str(p))
                    #print("flow " + str(pp))
                    d = dist_2d(pp, p)
                    if d < closest:
                        closest = d
                        index = k
                newVert.append(verts2[index])
                banned.append(index)
        # calculate the 3D positions for each point
        pts = []
        for i in range(0, len(verts1)):
            V1 = verts1[i]
            V2 = newVert[i]
            temp = np.copy(keypoints1)
            cv2.circle(temp, (V1[1], V1[0]), 3,  (255, 0, 255), 3)
            cv2.circle(temp, (V2[1], V2[0]), 3,  (0, 255, 0), 3)
            cv2.imwrite("views/" + self.toString() + "_view" + str(self.viewCount) + "_corr" + str(i) + ".png", img=temp)
            #invert the arrays
            p_cam, zdist = self.calculate_pos(V1[::-1], V2[::-1], b, reso=(1920 - 600, 1080-400), debug=True)  # in camera coordinates
            # ignore point if too far away to be accurate

            if (zdist < self.cameraPos["r"] + 0.5):
                p_world = self.cam.point2world(p_cam)
                p_cam = np.round(p_cam, 8)
                p_world = np.round(p_world, 8)
                pts.append(p_world)
        self.poly.addPoints(pts)
        return pts
        #self.poly.viewPoints()

    # shrinks image, less computations needed
    def getROI(self):
        self.cam.view = self.cam.view[200:1080 - 200, 300: 1920 - 300]
        self.resolution = (1920 - 600, 1080 - 400)

    #re-converts ROI coordinate point to 1920/1080 point coord
    def ROI_to_full(self, point):
        point[0] = point[0] + 300
        point[1] = point[1] + 200
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

def TestFindFeatures(robot):
    print(">>>> Testing feature correspondance")
    #robot.set_radius(2.5)
    #robot.calcViewFeatures()
    #robot.set_radius(2.6)
    #robot.calcViewFeatures()
    #robot.set_radius(2.7)
    #robot.calcViewFeatures()
    #robot.set_radius(2.8)
    #robot.calcViewFeatures()
    robot.set_radius(2.5)
    robot.calcViewFeatures('2.5')
    robot.set_radius(2.6)
    robot.calcViewFeatures('2.6')
    robot.set_radius(2.8)
    robot.calcViewFeatures('2.8')
    robot.set_radius(3)
    robot.calcViewFeatures('3')
    robot.set_radius(3.2)
    robot.calcViewFeatures('3.2')

    pts = (np.copy(robot.poly.points)).tolist()

    clusters = robot.getClusters(pts=pts, maxDist=0.04)
    points3d = []
    for C in clusters:
        points3d.append(WLSQ_3D(C))

    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(points3d)

    open3d.visualization.draw_geometries([pcl])

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR - SameDiff.py requires 1 argument <csvpath>")
        exit(0)

    if os.path.exists("views/movement.txt"):
        os.remove("views/movement.txt")

    file = str(sys.argv[1])  # Can replace this with the path of the file
    DL = DataLoader(file)

    robot1 = Robot(DL, 1, 0, debug=False)
    robot2 = Robot(DL, 1, 1, debug=True)



    #TestCamera2World(robot2)
    #TestCenterObject(robot2)
    #TestPointDepth(robot2)
    TestFindFeatures(robot2)
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

    robot2.poly.viewPoints()


