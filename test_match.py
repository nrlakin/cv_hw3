import numpy as np
import cv2
from matplotlib import pyplot as plt

def findMatches(template, image):
    sift = cv2.SIFT()
    kp_t, des_t = sift.detectAndCompute(template,None)
    kp_i, des_i = sift.detectAndCompute(image,None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_t, des_i, k=2)

    good = []
    for m,n in matches:
        if m.distance < .75*n.distance:
            good.append(m)

    """
    result = np.zeros(shape =(max([template.shape[0],image.shape[0]]), template.shape[1]+image.shape[1],3), dtype=np.uint8)
    result[:template.shape[0], :template.shape[1],0] = template
    result[:image.shape[0], template.shape[1]:, 0] = image
    result[:,:,1]=result[:,:,0]
    result[:,:,2]=result[:,:,0]

    for m in good:
        color = tuple([np.random.randint(0,255) for _ in xrange(3)])
        cv2.line(result, (int(kp_t[m.queryIdx].pt[0]), int(kp_t[m.queryIdx].pt[1])) , (int(kp_i[m.trainIdx].pt[0] + template.shape[1]), int(kp_i[m.trainIdx].pt[1])), color)

    points = []
    for m in good:
        points.append(kp_i[m.trainIdx].pt)

    points = np.array(points,dtype='int')
    min_x=np.min(points[:,0])
    max_x=np.max(points[:,0])
    min_y=np.min(points[:,1])
    max_y=np.max(points[:,1])
    cv2.rectangle(result,tuple([min_x+template.shape[1],min_y]),tuple([max_x+template.shape[1],max_y]), color)

    """
    result = np.zeros(shape = (image.shape[0],image.shape[1],3),dtype=np.uint8)
    result[:,:,0]=image
    result[:,:,1]=image
    result[:,:,2]=image

    points = []
    for m in good:
        points.append(kp_i[m.trainIdx].pt)

    points = np.array(points,dtype='int')
    min_x=np.min(points[:,0])
    max_x=np.max(points[:,0])
    min_y=np.min(points[:,1])
    max_y=np.max(points[:,1])
    color = tuple([np.random.randint(0,255) for _ in xrange(3)])
    #br = cv2.boundingRect(cv2.fromArray(points))
    cv2.rectangle(result,tuple([min_x,min_y]),tuple([max_x,max_y]), color, thickness=5)
    plt.imshow(result)
    plt.show()
