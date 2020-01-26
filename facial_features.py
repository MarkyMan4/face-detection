import cv2
import numpy as np


# lbp cascades are faster that haar
lbp_face = 'cascades/lbpcascade_frontalface.xml'
# haar_face = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(lbp_face)

haar_eye = 'cascades/haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(haar_eye)

haar_smile = 'cascades/haarcascade_smile.xml'
smile_cascade = cv2.CascadeClassifier(haar_smile)

# Detect faces, eyes and smiles. Draw boxes around them and label them
def find_features(frame, gray):
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
		cv2.rectangle(frame, (x, y-30), (x+70, y), (0, 0, 255), -1)
		cv2.putText(frame, 'face', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

		face_gray = gray[y : y + h, x : x + w]
		face_frame = frame[y : y + h, x : x + w]

		# detect eyes within face
		eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)
		
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(face_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
			cv2.rectangle(face_frame, (ex, ey - 15), (ex + 27, ey), (0, 255, 0), -1)
			cv2.putText(face_frame, 'eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

		# detect mouth/smiles within face
		smiles = smile_cascade.detectMultiScale(face_gray, 1.1, 150)
		
		for (sx, sy, sw, sh) in smiles:
			cv2.rectangle(face_frame, (sx, sy), (sx + sw, sy + sh), (255, 150, 0), 1)
			cv2.rectangle(face_frame, (sx, sy - 15), (sx + 45, sy), (255, 150, 0), -1)
			cv2.putText(face_frame, 'smile', (sx, sy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def main_loop(cap):
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			find_features(frame, gray)

			# used for detecting contours in image
			# edged = cv2.Canny(gray, 30, 200) 
			# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
			# cv2.drawContours(frame, contours, -1, (0, 255, 0), 3) 

			cv2.imshow('Detection', frame) 

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

def finish(cap):
	cap.release()
	cv2.destroyAllWindows()

def main():
	cap = cv2.VideoCapture(0)
	# print(cap.read())
	print(cap.isOpened())

	cap.set(3, 800)
	cap.set(4, 800)

	main_loop(cap)
	finish(cap)

if __name__ == '__main__':
	main()