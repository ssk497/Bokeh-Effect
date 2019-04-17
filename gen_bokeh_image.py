import cv2
import sys
import numpy as np
import scipy.signal

import face_detection
import closed_form_matting


def im2double(im):
	min_val = np.min(im.ravel())
	max_val = np.max(im.ravel())
	out = (im.astype('float') -min_val) / (max_val-min_val)
	return out


def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140][::-1])   


def generate_bokeh_image(img_path):
	blur_sigma = 1
	dBlur_thresh = 3
	upSample_type = 'linear'

	# Load Image
	img_orig = cv2.imread(img_path)

	# Resize Image
	nC = img_orig.shape[1]
	img_scale = min(1.0, 400.0/nC)
	width, height = img_orig.shape[1]*img_scale, img_orig.shape[0]*img_scale
	img_orig = cv2.resize(img_orig, (int(width), int(height)))
	cv2.imshow('Original Image After Resizing', img_orig)
	cv2.waitKey()

	# Process Image
	img = rgb2gray(im2double(img_orig))
	
	# Get original image.
	ix = img
	
	# Get re-blurred image
	ixx = cv2.GaussianBlur(ix, (4*blur_sigma+1, 4*blur_sigma+1), blur_sigma)
	
	# Get gradient of original image
	gx_ix = cv2.Sobel(ix, cv2.CV_64F, 1, 0)
	gy_ix = cv2.Sobel(ix, cv2.CV_64F, 0, 1)
	gm_ix = np.sqrt(np.power(gx_ix, 2) + np.power(gy_ix, 2))

	# Get gradient of re-blurred image
	gx_ixx = cv2.Sobel(ixx, cv2.CV_64F, 1, 0)
	gy_ixx = cv2.Sobel(ixx, cv2.CV_64F, 0, 1)
	gm_ixx = np.sqrt(np.power(gx_ixx, 2) + np.power(gy_ixx, 2))

	# Get gradient ratio
	R = np.divide(gm_ix, gm_ixx, out=np.zeros_like(gm_ix), where=gm_ixx != 0)
	R = np.where(R<=1, 1, R)

	# Get defocus blur
	P = np.sqrt(np.square(R)-1)
	DB = np.divide(float(blur_sigma), P, where=P != 0)
	DB = np.where(R<=1, 0, DB)
	DB = scipy.signal.medfilt2d(DB.astype('float32'), 3)
	DB = np.where(DB>dBlur_thresh, dBlur_thresh, DB)

	# Get edges
	slice1copy = scipy.ndimage.imread(img_path, 0)
	slice1copy = scipy.ndimage.zoom(slice1copy, (img_scale, img_scale, 1))
	bw = cv2.Canny(slice1copy, 50, 100)
	bw = im2double(bw)
	cv2.imshow('Edge Detection', bw)
	cv2.waitKey()

	# Get defocus blur only along the edges
	DB_edge = np.multiply(DB, bw)

	# Remove outliers
	DBtemp = np.reshape(DB_edge, (-1))
	DBtemp = DBtemp[~np.isnan(DBtemp)]
	DBtemp = np.sort(DBtemp)
	thresh_DBedge = np.mean(DBtemp) + 6*np.std(DBtemp)
	DB_edge = np.where(DB_edge>thresh_DBedge, thresh_DBedge, DB_edge)
	DB_edge = (DB_edge - np.min(DB_edge))
	DB_edge = DB_edge / (np.max(DB_edge))
	cv2.imshow('Sparse Defcous Map', DB_edge)
	cv2.waitKey()

	# Apply bilateral filtering
	DB_edge_BF = cv2.bilateralFilter(DB_edge.astype(np.float32), 21, 5, 0.3)
	cv2.imshow('Sparse Defcous Map With Bilateral Filter', DB_edge_BF)
	cv2.waitKey()

	# Get Matting-Laplacian matrix
	print("Calculating Matting-Laplacian matrix")
	h, w = bw.shape[0], bw.shape[1]
	img_lap = (img_orig - np.min(img_orig)) / (np.max(img_orig) - np.min(img_orig))
	L = closed_form_matting.compute_laplacian(img_lap)
	
	# Get Full Defocus map
	l = 0.005
	bwReshape = np.reshape(bw, (h*w, 1))
	DBReshape = np.reshape(DB_edge_BF, (h*w, 1))
	DBReshape = np.where(np.isnan(DBReshape), 0, DBReshape)

	vec = np.reshape(bwReshape, (-1))
	o = np.arange(0, len(vec))
	D = scipy.sparse.csc_matrix((vec, (o, o)), shape=(len(vec), len(vec)))

	A = (L+l*D)
	B = l*((scipy.sparse.csc_matrix(D @ DBReshape)))
	DBss = scipy.sparse.linalg.spsolve(A,B)
	DBss = np.reshape(DBss, (h, w))
	DB_fullMP = DBss
	cv2.imshow('Full Defocus Map', DB_fullMP)
	cv2.waitKey()

	# Perform post processing
	step_count = 3
	step_interval = np.linspace(np.min(DB_fullMP), np.max(DB_fullMP), step_count)

	blur_sigma = [2, 5, 10]
	for i in range(step_count-1):
		blur_mask = np.zeros((h, w))
		blur_mask_a = np.where(DB_fullMP>=step_interval[i], 1, 0)
		blur_mask_b = np.where(DB_fullMP<step_interval[i+1], 1, 0)
		blur_mask = np.logical_and(blur_mask_a, blur_mask_b)
		blur_mask = blur_mask.astype('uint8')
		if i==0:
			section_after_blur = cv2.bitwise_and(img_orig, img_orig, mask=blur_mask)
			bokeh_img = section_after_blur
			cv2.imshow('Foreground Image', bokeh_img)
			cv2.waitKey()
		else:
			orig_img_after_blur = cv2.GaussianBlur(img_orig, (blur_sigma[1], blur_sigma[1]), blur_sigma[1], borderType = cv2.BORDER_REPLICATE)  
			section_after_blur = cv2.bitwise_and(orig_img_after_blur, orig_img_after_blur, mask=blur_mask)
			cv2.imshow('Background Image', section_after_blur)
			cv2.waitKey()
			bokeh_img = bokeh_img + section_after_blur

	cv2.imshow('Bokeh Image', bokeh_img)
	cv2.waitKey()

	# Face protection
	(num_faces, face_mask, inverse_face_mask) = face_detection.get_face(img_path)
	if num_faces != 0:
		face_1 = cv2.bitwise_and(img_orig, img_orig, mask=face_mask)
		not_face = cv2.bitwise_and(bokeh_img, bokeh_img, mask=inverse_face_mask)
		final_img = face_1 + not_face
	else:
		final_img = bokeh_img
	cv2.imshow('Bokeh Image After Face Protection', final_img)
	cv2.waitKey()
	cv2.destroyAllWindows()


if __name__=='__main__':
	args = sys.argv
	if len(args) > 1:
		img_path = args[1]
	else:
		img_path = 'images/Test image 5.png'
	generate_bokeh_image(img_path)
