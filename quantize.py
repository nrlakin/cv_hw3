import cv2
import numpy as np
import random
import test_match as tm
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

# Expects BGR Image Matrix
def saturateImage(image, amount):
	(h, w) = image.shape[:2]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image = image.reshape((h * w, 3))
	
	# Saturate images
	for i, pixel in enumerate(image):
		if pixel[1] + amount < 255:
			pixel[1] += amount
		else:
			pixel[1] = 255
		image[i] = pixel

	image = image.reshape((h, w, 3))
	return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


# Expects BGR Image Matrix
# Returns RGB Image, LAB Clusters
def calculateQuantizedPalette(image, clusters):
	(h, w) = image.shape[:2]	
	image_qnt = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	image_qnt = image_qnt.reshape((h * w, 3))

	clt = KMeans(n_clusters = clusters)
	labels = clt.fit_predict(image_qnt)
	image_qnt = clt.cluster_centers_.astype("uint8")[labels]
	 
	# reshape the feature vectors to images
	image_qnt = image_qnt.reshape((h, w, 3))
	image_qnt = cv2.cvtColor(image_qnt, cv2.COLOR_LAB2BGR)

	return image_qnt, clt.cluster_centers_


def quantizeImage(image, clusters, cluster_threshold):
	(h, w) = image.shape[:2]	
	img_qnt = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	img_qnt = img_qnt.reshape((h * w, 3))

	# clahe = cv2.createCLAHE(clipLimit=1.)
	# img_qnt = clahe.apply(img_qnt)
	# img_qnt[:,0] = cv2.equalizeHist(img_qnt[:, 0])[:,0]

	for i, pixel in enumerate(img_qnt):
		dists = np.sqrt(np.sum((clusters - pixel)**2, axis=1))
		nearest = np.argmin(dists)
		if dists[nearest] > cluster_threshold:
			pixel = [0, 127, 127]
		else:
			pixel = clusters[nearest]
		img_qnt[i] = pixel


	# reshape the feature vectors to images
	img_qnt = img_qnt.reshape((h, w, 3))
	return cv2.cvtColor(img_qnt, cv2.COLOR_LAB2BGR)


# Main
if __name__ == '__main__':

	# Variables
	saturation = 30
	max_template_width = 400.0
	max_image_width = 800.0
	cluster_threshold = 50
	cluster_count = 4

	# Load images (resize template if necessary)
	template = cv2.imread('Superman/template.png')
	if template.shape[1] > max_template_width:
		scale = max_template_width/template.shape[1]
		template = cv2.resize(template, (0,0), fx=scale, fy=scale)
	img_bgr = cv2.imread('Superman/image3.jpg')
	if img_bgr.shape[1] > max_image_width:
		scale = max_image_width/img_bgr.shape[1]
		img_bgr = cv2.resize(img_bgr, (0,0), fx=scale, fy=scale)

	# Get color palette
	template_qnt = saturateImage(template, saturation)
	template_qnt, clusters = calculateQuantizedPalette(template_qnt, cluster_count)
	cv2.imshow("image1", np.hstack([template, template_qnt]))

	# Color Quantization
	(h, w) = img_bgr.shape[:2]
	img_qnt = saturateImage(img_bgr, saturation)
	img_qnt = quantizeImage(img_qnt, clusters, cluster_threshold)

	cv2.imshow("image2", np.hstack([img_bgr, img_qnt]))
	cv2.waitKey(0)

	# img_qnt_gray = cv2.cvtColor(img_qnt, cv2.COLOR_BGR2GRAY)
	# template_gray = cv2.cvtColor(template_qnt, cv2.COLOR_BGR2GRAY)
	# img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

	# tm.findMatches(template_gray, img_qnt_gray)
	# tm.findMatches(template_gray, img_gray)



