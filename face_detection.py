import cv2 
import numpy as np

def get_face(img_path):
	img_orig = cv2.imread(img_path)
	nC = img_orig.shape[1]
	img_scale = min(1.0, 400.0/nC)
	width, height = img_orig.shape[1]*img_scale, img_orig.shape[0]*img_scale
	img = cv2.resize(img_orig, (int(width), int(height)))
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	mask = None
	face_mask = None
	
	for (x, y, w, h) in faces:
		face_mask = cv2.rectangle(np.zeros((img.shape[0], img.shape[1])), (x, y-int(0.3*h)), (x+w, y+h), (255, 0, 0), cv2.FILLED)
		img = cv2.rectangle(img, (x, y-int(0.3*h)), (x+w, y+h), (255, 0, 0), 2)
		roi_gray = gray[y-int(0.3*h):y+h, x:x+w]
		roi_color = img[y-int(0.3*h):y+h, x:x+w]
	
	if len(faces) != 0:
		face_mask = np.where(face_mask==255, 1, 0)
		inverse_face_mask = np.where(face_mask==1, 0, 1)
		return (len(faces), face_mask.astype("uint8"), inverse_face_mask.astype("uint8"))
	
	return (0, None, None)
