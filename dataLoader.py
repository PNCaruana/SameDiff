from websocket import create_connection
import io, sys, json, base64
from json import dumps
from PIL import Image
import cv2
import numpy as np
import csv


class DataLoader:
    def __init__(self, csvpath):
        self.objects = []
        with open(csvpath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.objects.append(row)
        print(self.objects)


    # returns cv2 image array
    def getView(self, pairNo, objNo, params, isLightFixed='true', isRandomCam='false'):

        ws = create_connection("wss://polyhedral.eecs.yorku.ca/api/")

        parameter = {
            'ID': self.objects[pairNo][objNo],
            'light_fixed': isLightFixed,
            'random_cam': isRandomCam,
            'cam_x': params["cam_x"],
            'cam_y': params["cam_y"],
            'cam_z': params["cam_z"],
            'cam_qw': params["cam_qw"],
            'cam_qx': params["cam_qx"],
            'cam_qy': params["cam_qy"],
            'cam_qz': params["cam_qz"]
        }

        json_params = dumps(parameter, indent=2)

        # send API request
        ws.send(json_params)

        while True:
            result = json.loads(ws.recv())
            print("Job Status: {0}".format(result['status']))
            if result['status'] == "SUCCESS":
                break
            elif "FAILURE" in result['status'] or "INVALID" in result['status']:
                sys.exit()

        # process result

        image_base64 = result['image']
        image_decoded = base64.b64decode(str(image_base64))

        # create cv2 image

        image = Image.open(io.BytesIO(image_decoded))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        ws.close()

        return cv_image
