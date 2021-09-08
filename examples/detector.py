# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

# import sys, os
# sys.path.append(os.path.join(os.getcwd(),'python/'))

# import darknet as dn
# import pdb

# dn.set_gpu(0)
# net = dn.load_net("cfg/yolo-thor.cfg", "/home/pjreddie/backup/yolo-thor_final.weights", 0)
# meta = dn.load_meta("cfg/thor.data")
# r = dn.detect(net, meta, "data/bedroom.jpg")
# print(r)

# # And then down here you could detect a lot more images like:
# r = dn.detect(net, meta, "data/eagle.jpg")
# print(r)
# r = dn.detect(net, meta, "data/giraffe.jpg")
# print(r)
# r = dn.detect(net, meta, "data/horses.jpg")
# print(r)
# r = dn.detect(net, meta, "data/person.jpg")
# print(r)

import numpy as np
import time
import cv2


INPUT_FILE1='/Users/User-PC/RA/pjreddie/path1.txt'
INPUT_FILE2='/Users/User-PC/RA/pjreddie/path2.txt'
#OUTPUT_FILE='predicted.jpg'
LABELS_FILE='/Users/User-PC/RA/pjreddie/data/coco.names'
CONFIG_FILE='/Users/User-PC/RA/pjreddie/cfg/yolov3.cfg'
WEIGHTS_FILE='/Users/User-PC/RA/pjreddie/yolov3.weights'
CONFIDENCE_THRESHOLD=0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

queue = []
with open(INPUT_FILE1, 'r') as f:
	lines = f.read().splitlines()
	queue.append(lines)

with open(INPUT_FILE2, 'r') as f:
	lines = f.read().splitlines()
	queue.append(lines)

length = len(lines)
m = 0
k = 0
count = 0

while m < length:
	image = cv2.imread(queue[k][m])
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
		CONFIDENCE_THRESHOLD)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]

			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)
	
	if k == 0:
		k = k + 1
	else:
		k = k - 1
		m = m + 1
	count = 10*k + m
	cv2.imwrite(f'/Users/User-PC/RA/pjreddie/results/image{count}.jpg', image)

	