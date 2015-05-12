import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCHES=6

def drawMatches(template, image, kp_t, kp_i, matches):
    result = np.zeros(shape =(max([template.shape[0],image.shape[0]]), template.shape[1]+image.shape[1],3), dtype=np.uint8)
    result[:image.shape[0], :image.shape[1],0] = image
    result[:template.shape[0], image.shape[1]:, 0] = template
    result[:,:,1]=result[:,:,0]
    result[:,:,2]=result[:,:,0]

    for m in matches:
        color = tuple([np.random.randint(0,255) for _ in xrange(3)])
        cv2.line(result, (int(kp_i[m.queryIdx].pt[0]), int(kp_i[m.queryIdx].pt[1])) , (int(kp_t[m.trainIdx].pt[0] + template.shape[1]), int(kp_t[m.trainIdx].pt[1])), color)

    plt.imshow(result)
    plt.show()

def findMatches(template, image):
    """
    Takes grayscale template and image, finds SIFT matches, draws a box around them.
    """
    sift = cv2.SIFT()
    #sift = cv2.SURF()
    kp_t, des_t = sift.detectAndCompute(template,None)
    kp_i, des_i = sift.detectAndCompute(image,None)

    print "keypoints in image: " + str(len(kp_i))
    print "keypoints in template: " + str(len(kp_t))
    matcher = cv2.BFMatcher()
    # tricky--flip query image and template here
    matches = matcher.knnMatch(des_i, des_t, k=2)

    good = []
    for m,n in matches:
        if m.distance < .5*n.distance:
            good.append(m)

    return good, kp_t, kp_i

def findObjects(template, image, kp_t, kp_i, matches):
    plt.gray()

    n_points = MIN_MATCHES
    h, w = template.shape
    result = np.zeros(shape = (image.shape[0],image.shape[1],3),dtype=np.uint8)
    result[:,:,0]=image
    result[:,:,1]=result[:,:,0]
    result[:,:,2]=result[:,:,0]

    obj_points = np.array([kp_t[m.trainIdx].pt for m in matches])
    scene_points = np.array([kp_i[m.queryIdx].pt for m in matches])
    corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

    while(n_points >= MIN_MATCHES):
        if obj_points.shape[0] < MIN_MATCHES:
            break
        H, mask = cv2.findHomography(obj_points, scene_points, cv2.RANSAC, ransacReprojThreshold=3)
        print H
        n_points = np.count_nonzero(mask)
        if n_points < MIN_MATCHES:
            break
        object = cv2.perspectiveTransform(corners, H).astype(np.int)
        #cv2.polylines(result, object, True, (0,255,0), 4, lineType=cv2.CV_AA)
        cv2.line(result, (object[0][0][0],object[0][0][1]),(object[1][0][0],object[1][0][1]), (0,255,0), thickness=4)
        cv2.line(result, (object[1][0][0],object[1][0][1]),(object[2][0][0],object[2][0][1]), (0,255,0), thickness=4)
        cv2.line(result, (object[2][0][0],object[2][0][1]),(object[3][0][0],object[3][0][1]), (0,255,0), thickness=4)
        cv2.line(result, (object[3][0][0],object[3][0][1]),(object[0][0][0],object[0][0][1]), (0,255,0), thickness=4)
        indices = np.where(mask==0)[0]
        obj_points = obj_points[indices]
        scene_points = scene_points[indices]
        # plt.imshow(result)
        # plt.show()

    plt.imshow(result)
    plt.show()

    return result




def drawObject(template, image, good):
    result = np.zeros(shape = (image.shape[0],image.shape[1],3),dtype=np.uint8)
    result[:,:,0]=image
    result[:,:,1]=image
    result[:,:,2]=image

    points = []
    for m in good:
        points.append(kp_i[m.trainIdx].pt)
        #result[kp_i[m.trainIdx].pt[1],kp_i[m.trainIdx].pt[0]]+=1
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

if __name__ == "__main__":
    plt.gray()

    #stub = 'superman/'
    stub = ''
    template = cv2.imread(stub + 'template.jpg',0)
    for i in range(1,10):
        im = cv2.imread(stub + 'image' + str(i) + '.jpg', 0)
        matches, kpt, kpi = findMatches(template, im)
        result = findObjects(template, im, kpt, kpi, matches)
        fig, ax = plt.subplots(1,1)
        ax.imshow(result)
        fig.show()
    input('enter to exit.')
