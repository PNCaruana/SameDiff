import csv
import os
import bpy

filename = "P:/Documents/Programming/Repos/SameDiff/views/movement.txt"

scene = bpy.data.scenes["Scene"]
scene.render.resolution_x = 480
scene.render.resolution_y = 480
scene.camera.rotation_mode = "QUATERNION"

cam = bpy.data.objects["Camera"]
cam.animation_data_clear()

with open(filename, 'r', newline='\n') as csvfile:
    ofile = csv.reader(csvfile, delimiter=',')
    
    rows = (r for r in ofile if r)
    
    coords = [[float(i) for i in r] for r in rows]
    
frame = 0
for c in coords:
    scene.camera.rotation_quaternion[0] = c[3]
    scene.camera.rotation_quaternion[1] = c[4]
    scene.camera.rotation_quaternion[2] = c[5]
    scene.camera.rotation_quaternion[3] = c[6]
    
    scene.camera.location.x = c[0]
    scene.camera.location.y = c[1]
    scene.camera.location.z = c[2]
    
    cam.keyframe_insert(data_path="location", frame = frame)
    cam.keyframe_insert(data_path="rotation_quaternion", frame = frame)
    frame += 5