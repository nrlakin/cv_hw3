import cv2
import numpy as np
import random
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

# Main
if __name__ == '__main__':

	# Load images (resize template if necessary)
	max_width = 400.0
	template = cv2.imread('roadsign/template.jpg')
	if template.shape[1] > max_width:
		scale = max_width/template.shape[1]
		template = cv2.resize(template, (0,0), fx=scale, fy=scale)
	# template = np.asarray(template, np.float32)
	img_bgr = cv2.imread('roadsign/image4.jpg')
	# img_bgr = np.asarray(img_bgr, np.float32)

	# Color Quantization
	(h, w) = template.shape[:2]
	template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
	template_lab = template_lab.reshape((template_lab.shape[0] * template_lab.shape[1], 3))
	clt = KMeans(n_clusters = 4)

	labels = clt.fit_predict(template_lab)
	template_qnt = clt.cluster_centers_.astype("uint8")[labels]
	 
	# reshape the feature vectors to images
	template_qnt = template_qnt.reshape((h, w, 3))
	template_lab = template_lab.reshape((h, w, 3))
	template_qnt = cv2.cvtColor(template_qnt, cv2.COLOR_LAB2BGR)


	cv2.imshow("image1", np.hstack([template, template_qnt]))
	# # cv2.waitKey(0)

	# Color Quantization
	(h, w) = img_bgr.shape[:2]
	img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
	img_lab = img_lab.reshape((img_lab.shape[0] * img_lab.shape[1], 3))

	threshold = 50

	img_qnt = img_lab
	for i, pixel in enumerate(img_qnt):
		dists = np.sqrt(np.sum((clt.cluster_centers_ - pixel)**2, axis=1))
		nearest = np.argmin(dists)
		if dists[nearest] > threshold:
			pixel[0] = 0.
		else:
			pixel = clt.cluster_centers_[nearest]
		img_qnt[i] = pixel


	# reshape the feature vectors to images
	img_qnt = img_qnt.reshape((h, w, 3))
	img_lab = img_lab.reshape((h, w, 3))
	 
	# convert from L*a*b* to RGB
	img_qnt = cv2.cvtColor(img_qnt, cv2.COLOR_LAB2BGR)

	cv2.imshow("image2", np.hstack([img_bgr, img_qnt]))
	cv2.waitKey(0)





