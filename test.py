# python test.py --shape-predictor shape_predictor_68_face_landmarks.dat --detector face_detection_model --alarm alarm.wav

# import the necessary packages
import numpy as np
import playsound
import argparse
import imutils
import time
import cv2
import os
import dlib
import requests
from imutils.video import FPS, VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread
from PySide6.QtWidgets import  QApplication, QWidget, QLabel, QSizePolicy, QGridLayout
from PySide6.QtGui import QMovie
from PySide6.QtCore import QSize, Qt
import queue as q

SONNO = "4-NOLOOP-Sonno.gif"
NOIA = "3-LOOP-Noia.gif"
INTERESSE = "1-LOOP-interesse-no-occhiaie.gif"
OCCHIAIE = "2-LOOP-interesse-occhiaie.gif"

controller = 0

stato = "Interesse"

class Windows(QWidget):
	def __init__(self):
		super().__init__()
		self.setGeometry(0,0,1920,1080)
		self.setWindowTitle("Maestrale Machine Learning")
		self.label = QLabel(self)
		self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.label.setAlignment(Qt.AlignCenter)
		self.label.setStyleSheet("QLabel {background-color: grey;}")
		self.layout = QGridLayout()
		self.layout.addWidget(self.label, 0, 0)
		self.setLayout(self.layout)

		self.changeMovie(INTERESSE)

	def changeMovie(self, gifMovie):
		self.movie = QMovie(gifMovie)
		self.movie.setScaledSize(QSize().scaled(1920, 800, Qt.KeepAspectRatio))
		self.label.setMovie(self.movie)
		self.movie.start()

app = QApplication([])
window = Windows()
window.show()

queue = q.Queue()
starttime = time.time()
executeApi = True
url = "http://localhost:5000/presentazione"

def httpRequest():
	while executeApi:
		try:
			response = requests.request("GET", url)
			queue.put(response.json()["data"])
		except:
			queue.put(0)
		time.sleep(1.0 - ((time.time() - starttime) % 1.0))


apiRequest = Thread(target=httpRequest)
apiRequest.deamon = True
apiRequest.start()

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
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
args = vars(ap.parse_args())


#------EYE------
# eye aspect ration threesold
EYE_AR_THRESH = 0.23
#E se occhi chiusi per 35 frame possibile dormita
EYE_AR_CONSEC_FRAMES = 35
#E se occhi chiusi per 100 frame è sicuramente addormentato
EYE_AR_CONSEC_FRAMES_DANGER = 80

#------MOUTH------
# MOUTH aspect ration threesold
MOUTH_AR_THRESH = 0.1
SLEEP_MOUTH_AR_THRESH = 0.02
#M se la bocca è aperta per 30 frame con un tresh di 7 è uno sbadiglio
MOUTH_AR_CONSEC_FRAMES = 20
#ms se la bocca è aperta per 100 frame con un tresh di 5 è un rilassamento della mascella
SLEEP_MOUTH_AR_CONSEC_FRAMES = 80

#-----Count of
DANGER_ZONE=0
DANGER_ZONE_MAX=300

MAX_FACES = 1
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off

MCOUNTER = np.zeros(MAX_FACES)
SLEEP_MOUTH_COUNTER = np.zeros(MAX_FACES)
ALARM_ON = np.zeros(MAX_FACES, dtype=bool)
COUNTER = np.zeros(MAX_FACES)


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
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detectorFaceRecognition = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# start the video stream thread
print("Starting the video stream thread")


vs = VideoStream(src=0).start()
fileStream = False

time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()
key = cv2.waitKey(1) & 0xFF
# loop over frames from the video file stream
while key != ord("q"):

	if(not queue.empty()):
		controller = queue.get()

	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=1000)

	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detectorSleep(gray, 0)

	# loop over the face detections
	for i, rect in enumerate(rects):
		
		if (i < MAX_FACES):
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

			# for (x, y) in shape:
			# 	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			# check to see if the mouth aspect ratio is over...
			if mouthMAR > MOUTH_AR_THRESH:
				MCOUNTER[i] += 1

				# if the mouth were closed for a sufficient number of frame...SBADIGLIO
				if MCOUNTER[i] >= MOUTH_AR_CONSEC_FRAMES and ear < 0.3:
					cv2.putText(frame, "SBADIGLIO", (shape[0][0], shape[49][1]),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
					DANGER_ZONE=DANGER_ZONE_MAX
					if (controller == 0 and stato != "noia" and stato != "sonno"):
						stato = "noia"
						window.changeMovie(NOIA)

			else:
				MCOUNTER[i] = 0

			# check to see if the mouth aspect ratio is over...MASCELLA RILASSATA
			if mouthMAR > SLEEP_MOUTH_AR_THRESH:
				SLEEP_MOUTH_COUNTER[i] += 1

				# if the mouth were closed for a sufficient number of...
				if SLEEP_MOUTH_COUNTER[i] >= SLEEP_MOUTH_AR_CONSEC_FRAMES:
					cv2.putText(frame, "RELAX", (shape[0][0], shape[57][1]),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 2)
					if DANGER_ZONE < DANGER_ZONE_MAX:
						if DANGER_ZONE < 0:
							DANGER_ZONE = 0
						DANGER_ZONE+=4
						if (controller == 0 and stato != "noia" and stato != "sonno"):
							stato = "noia"
							window.changeMovie(NOIA)

			else:
				SLEEP_MOUTH_COUNTER[i] = 0


			if ear < EYE_AR_THRESH:
				COUNTER[i] += 1
				if DANGER_ZONE:
					COUNTER[i]+=1

				# if the eyes were closed for a sufficient number of frame
				# then write alert
				if COUNTER[i] >= EYE_AR_CONSEC_FRAMES:
					cv2.putText(frame, "APRI GLI OCCHI", (shape[0][0], shape[42][1]),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

				# if the eyes were closed for a sufficient number of frame
				# then enter in danger zone and play alarm sound
				if COUNTER[i] >= EYE_AR_CONSEC_FRAMES_DANGER:
					# # if the alarm is not on, turn it on
					# if not ALARM_ON[i]:
					# 	ALARM_ON[i] = True

					# 	# check to see if an alarm file was supplied,
					# 	# and if so, start a thread to have the alarm
					# 	# sound played in the background
					# 	if args["alarm"] != "":
					# 		t = Thread(target=sound_alarm,
					# 			args=(args["alarm"],))
					# 		t.deamon = True
					# 		t.start()

					# draw an alarm on the frame
					cv2.putText(frame, "SVEGLIATI!", (shape[0][0], shape[22][1]),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					DANGER_ZONE=DANGER_ZONE_MAX
					if (controller == 0 and stato != "sonno"):	
						stato = "sonno"
						window.changeMovie(SONNO)

			# otherwise, the eye aspect ratio is not below the blink
			# threshold, so reset the counter and alarm
			else:
				if (ear > 0.4):
					DANGER_ZONE-=2
				DANGER_ZONE-=2
				COUNTER[i] = 0
				ALARM_ON[i] = False
				if (DANGER_ZONE <= 0):	
					if (controller == 0 and stato != "interesse" and stato == "occhiaie"):
						stato = "interesse"
						window.changeMovie(INTERESSE)
					if (controller == 0 and stato != "occhiaie" and (stato == "sonno" or stato == "noia")):
						stato = "occhiaie"
						window.changeMovie(OCCHIAIE)
						DANGER_ZONE = DANGER_ZONE_MAX

			# draw the computed eye aspect ratio on the frame to help
			# with debugging and setting the correct eye aspect ratio
			# thresholds and frame counters
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "MAR: {:.2f}".format(mouthMAR), (250, 30),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "E: {:.2f}".format(COUNTER[0]), (30, 420),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "M: {:.2f}".format(MCOUNTER[0]), (250, 420),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "MS: {:.2f}".format(SLEEP_MOUTH_COUNTER[0]), (490, 420),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Dan: {:.2f}".format(DANGER_ZONE), (440, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Stato: {0}".format(str(stato)), (440, 70),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	if(controller != 0):
		if (controller == 1 and stato != "interesse"):
			stato = "interesse"
			window.changeMovie(INTERESSE)
		if (controller == 2 and stato != "noia"):
			stato = "noia"
			window.changeMovie(NOIA)
		if (controller == 3 and stato != "occhiaie"):
			stato = "occhiaie"
			window.changeMovie(OCCHIAIE)
		if (controller == 4 and stato != "sonno"):
			stato = "sonno"
			window.changeMovie(SONNO)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF




executeApi=False
# stop the timer and display FPS information
fps.stop()
print("Time: {:.2f}".format(fps.elapsed()))
print("FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
