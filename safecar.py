# USAGE

#TRAIN
#python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

#python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle 	--le output/le.pickle

# python safecar.py --shape-predictor shape_predictor_68_face_landmarks.dat --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle  --alarm alarm.wav

# import the necessary packages
import numpy as np
import playsound
import argparse
import imutils
import pickle
import time
import cv2
import os
import dlib
from imutils.video import FPS, VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
 	A = dist.euclidean(mouth[14], mouth[18])
 	B = dist.euclidean(mouth[0], mouth[6])
	# compute the mouth aspect ratio
 	mar = (A)/(B)
	# return the mouth aspect ratio
 	return mar
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file") 
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
args = vars(ap.parse_args())
 

#------EYE------
# eye aspect ration threesold
EYE_AR_THRESH = 0.3
#E se occhi chiusi per 35 frame possibile dormita
EYE_AR_CONSEC_FRAMES = 35
#E se occhi chiusi per 100 frame è sicuramente addormentato
EYE_AR_CONSEC_FRAMES_DANGER = 80

#------MOUTH------
# MOUTH aspect ration threesold
MOUTH_AR_THRESH = 0.1
SLEEP_MOUTH_AR_THRESH = 4
#M se la bocca è aperta per 30 frame con un tresh di 7 è uno sbadiglio
MOUTH_AR_CONSEC_FRAMES = 20
#ms se la bocca è aperta per 100 frame con un tresh di 5 è un rilassamento della mascella
SLEEP_MOUTH_AR_CONSEC_FRAMES = 80

#-----Count of 
DANGER_ZONE=0
DANGER_ZONE_MAX=1000


# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
MCOUNTER = 0
SLEEP_MOUTH_COUNTER = 0
ALARM_ON = False
COUNTER = 0
LeftCounter = 0
RightCounter = 0
TOTAL = 0
LeftTotal = 0
RightTotal = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("Loading the facial landmark predictor")
detectorSleep = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# load our serialized face detector from disk
print("Loading the face detector")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detectorFaceRecognition = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("Loading the face recognizer")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# start the video stream thread
print("Starting the video stream thread")


# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

vs = VideoStream(src=0).start()
fileStream = False

time.sleep(1.0)



# start the FPS throughput estimator
fps = FPS().start()
key = cv2.waitKey(1) & 0xFF
# loop over frames from the video file stream
while key != ord("q"):
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detectorFaceRecognition.setInput(imageBlob)
	detections = detectorFaceRecognition.forward()
	
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			if proba < args["confidence"]:
				cv2.putText(frame, "Antifurto", (200, 250),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		


	# update the FPS counter
	fps.update()


	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detectorSleep(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		
		
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouthMAR = mouth_aspect_ratio(mouth)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)
		
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# check to see if the mouth aspect ratio is over...
		if mouthMAR > MOUTH_AR_THRESH:
			MCOUNTER += 1

			# if the mouth were closed for a sufficient number of frame...SBADIGLIO
			if MCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
				cv2.putText(frame, "SBADIGLIO", (50, 150),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
				DANGER_ZONE=DANGER_ZONE_MAX

		else:
			MCOUNTER = 0

		# check to see if the mouth aspect ratio is over...MASCELLA RILASSATA
		if mouthMAR > SLEEP_MOUTH_AR_THRESH:
			SLEEP_MOUTH_COUNTER += 1

			# if the mouth were closed for a sufficient number of...
			if SLEEP_MOUTH_COUNTER >= SLEEP_MOUTH_AR_CONSEC_FRAMES:
				cv2.putText(frame, "RELAX", (350, 150),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2)
				DANGER_ZONE=DANGER_ZONE_MAX

		else:
			SLEEP_MOUTH_COUNTER = 0


		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if DANGER_ZONE:
				COUNTER+=1

			# if the eyes were closed for a sufficient number of frame
			# then write alert
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				cv2.putText(frame, "APRI GLI OCCHI", (50, 300),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 200), 2)

			# if the eyes were closed for a sufficient number of frame
			# then enter in danger zone and play alarm sound
			if COUNTER >= EYE_AR_CONSEC_FRAMES_DANGER:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "SVEGLIATI!", (350, 300),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			DANGER_ZONE-=2
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mouthMAR), (250, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "E: {:.2f}".format(COUNTER), (30, 420),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "M: {:.2f}".format(MCOUNTER), (250, 420),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MS: {:.2f}".format(SLEEP_MOUTH_COUNTER), (490, 420),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Dan: {:.2f}".format(DANGER_ZONE), (440, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
 


# stop the timer and display FPS information
fps.stop()
print("Time: {:.2f}".format(fps.elapsed()))
print("FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()