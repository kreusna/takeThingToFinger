import cv2
import numpy as np

capture = cv2.VideoCapture(0)

getImage = cv2.imread('baby.jpeg')
smallImage = cv2.resize( getImage, (75, 75))

smallImageoldx = 400
smallImageoldy = 350

smallImageoldwidth = smallImageoldx + 75
smallImageoldhigh = smallImageoldy + 75

def calculateDistance(topLists):
    result = [0,0,0]
    if len(topLists) > 1:
        top1x = topLists[0][0]
        top2x = topLists[1][0]
        distance = top1x - top2x
        if top2x > top1x:
            distance = top2x - top1x

        distanceX = int(top1x/2) + int(top2x/2)
        distanceY = int(topLists[0][1]/2) + int(topLists[1][1]/2)  
        result = [distance,distanceX, distanceY]
    return result

def getContours(img):
    touch = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area >15000:
            topArea = []

            getHull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt,getHull)

            if len(getHull) > 3:
                end = ()
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    if d > 1500 and cnt[e][0][0] > 30 :
                        end = tuple(cnt[e][0])
                        topArea.append(cnt[e][0])
                        
                        cv2.circle(imageContour,end,5,[255,0,0],-1)
                        
            newTopdata = sorted(topArea, key=lambda x: x[1] , reverse=True) 
            getDistanceFromTopFinger = calculateDistance(newTopdata)
            if (getDistanceFromTopFinger[0] < 50):
                touch = 1

    global smallImageoldx
    global smallImageoldy
    global smallImageoldhigh
    global smallImageoldwidth
    if touch ==1 and getDistanceFromTopFinger[1] > 0 :
        smallImageoldx = getDistanceFromTopFinger[1] - 30
        smallImageoldy = getDistanceFromTopFinger[2] - 30
        smallImageoldhigh = smallImageoldy +75
        smallImageoldwidth = smallImageoldx + 75
    else:
        smallImageoldx = 400
        smallImageoldy = 350
        smallImageoldhigh = smallImageoldy +75
        smallImageoldwidth = smallImageoldx + 75
    imageContour[smallImageoldy:smallImageoldhigh, smallImageoldx:smallImageoldwidth] = smallImage

while True:
    _, frame = capture.read()
    imageContour = frame.copy()
    
    medianImage = cv2.medianBlur(frame,17)
    gaussianImage = cv2.GaussianBlur(medianImage,(5,5),0)
    bilateralImage = cv2.bilateralFilter(gaussianImage,9,75,75)
    imgHSV = cv2.cvtColor(bilateralImage, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 33, 35]) 
    upper = np.array([39, 186 , 145])

    mask = cv2.inRange(imgHSV, lower, upper)

    getContours(mask)

    cv2.imshow("video",imageContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()