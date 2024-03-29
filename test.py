from websocket import create_connection
import io, sys, json, base64
from json import dumps
from PIL import Image
import cv2
import numpy as np
import time

print("making connection")
# Create Connection
#ws = create_connection("wss://230t.eecs.yorku.ca:8044/api/")
ws = create_connection("wss://polyhedral.eecs.yorku.ca/api/")
print("connected")
# Set Parameters
parameter = {
    'ID':'b04b67e3-9c5c-4468-9241-56288d6509ed',
    'light_fixed':'true',
    'random_cam': 'true',
    'cam_x':-0.911,
    'cam_y':1.238,
    'cam_z':-4.1961,
    'cam_qw':-0.0544,
    'cam_qx':-0.307,
    'cam_qy':0.9355,
    'cam_qz':0.16599
}
json_params = dumps(parameter, indent=2)

# Send API request
ws.send(json_params)

# Wait patiently while checking status
timeout = 0
start = time.time();
while True:
    timeout = time.time()
    result = json.loads(ws.recv())
    print("Job Status: {0}".format(result['status']))
    if result['status'] == "SUCCESS":
        break
    elif "FAILURE" in result['status'] or "INVALID" in result['status']:
        sys.exit()

    print("Elapsed: " + str(timeout - start))

# Processing result
image_base64 = result['image']
image_decoded = base64.b64decode(str(image_base64))

# Create Open CV 2 Image
image = Image.open(io.BytesIO(image_decoded))
cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
cv2.imshow('image',cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Close Connection
ws.close()
