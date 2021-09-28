from PIL import ImageGrab
import numpy as np
import cv2
import math
from skimage.metrics import structural_similarity as ssim

def CapturingBoard():
    while(True):
        img = ImageGrab.grab(bbox=(0,0,900,900)) #bbox specifies specific region (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        cv2.imshow("Live Screen", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(filename='Screenshot.jpg', img=frame)
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def CroppingBoard():
    frame = cv2.imread("Screenshot.jpg")
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, (0, 0, 0), (179, 255, 160))
    img_not = cv2.bitwise_not(mask)

    contours, hierarchy = cv2.findContours(img_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    roi = frame[y: y+h, x: x+w]

    cv2.imwrite(filename="Initial_Image.jpg", img=roi)

    return x, y, w, h

def findSquareContoursOnline(CalibratingImg):
    print("You were here in the findSquareContoursOnline Function !!")
    chessBoardOriginal = cv2.imread(CalibratingImg)

    img = np.zeros(chessBoardOriginal.shape, dtype=np.uint8)

    side = chessBoardOriginal.shape[0]
    edge_length = (side / 8) - 1

    for i in range(0, 9):

        img = cv2.line(img, (0, int(i*edge_length)), (side, int(i*edge_length)), (255, 255, 255), thickness=1)
        img = cv2.line(img, (int(i*edge_length), 0), (int(i*edge_length), side), (255, 255, 255), thickness=1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10000:
            cnts.append(cnt)

    cv2.drawContours(chessBoardOriginal, cnts, -1, (0, 0, 255))
    cv2.drawContours(img, cnts, -1, (0, 0, 255))

    # cv2.imshow("Made Up Image", img)
    # cv2.imshow("Original Image", chessBoardOriginal)
    # cv2.waitKey(0)

    return cnts

def getChessSquares():

    contours = findSquareContoursOnline("Initial_Image.jpg")
    img = cv2.imread("Initial_Image.jpg")
    cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    
    def contoursortX(c):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        return cX

    def contoursortY(c):
        M = cv2.moments(c)
        cY = int(M["m01"] / M["m00"])
        return -cY

    number_to_alpha = {0 :"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}

    contours.sort(key= contoursortX)

    new_contours = []

    for i in range(8):
        alpha =  number_to_alpha[i]
        new_list = contours[8*i: 8*i + 8]
        new_list.sort(key = contoursortY)
        for i in range(8):
            new_contours.append([new_list[i], alpha + str(i + 1)])

    return new_contours

def pieceMovementOnline():
    imageA = cv2.imread("Initial_Image.jpg")
    imageB = cv2.imread("Final_Image.jpg")

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts.sort (key = lambda x: cv2.contourArea (x), reverse = True)
    max_cnt = cnts[0]
    x, y, w, h = cv2.boundingRect(max_cnt)

    max_cnt2 = cnts[1]
    a, b, c, d = cv2.boundingRect(max_cnt2)
    return(x, y, w, h), (a, b, c, d)

def getMoveSquaresOnline(dimensions, whiteToMove, board):
    while(True):
        img = ImageGrab.grab(bbox=(0,0,1000,1000)) #bbox specifies specific region (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        roi = frame[y: y+h, x: x+w]
        cv2.imwrite(filename="Final_Image.jpg", img= roi)
        cv2.destroyAllWindows()
        break

    end, start = pieceMovementOnline()

    p, q, r, s = end
    a, b, c, d = start
    endPnt = p+r//2, q+s//2
    startPnt = a+c//2, b+d//2

    startSq = "a1"
    endSq = "h8"

    for item in finalList2:
        cnt = item[0]
        if cv2.pointPolygonTest(cnt, endPnt, False) == 1.0 :
            endSq = item[1]
        if cv2.pointPolygonTest(cnt, startPnt, False) == 1.0 :
            startSq = item[1]

    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,"5": 3, "6": 2, "7": 1, "8": 0}
    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    startSqcol = filesToCols[startSq[0]]
    startSqrow = ranksToRows[startSq[1]]

    endSqcol = filesToCols[endSq[0]]
    endSqrow = ranksToRows[endSq[1]]

    if whiteToMove :
        if board[endSqrow][endSqcol][0] != "w":
            endSqcolf, endSqrowf = endSqcol, endSqrow
            startSqcolf, startSqrowf = startSqcol, startSqrow
        else :
            endSqcolf, endSqrowf = startSqcol, startSqrow
            startSqcolf, startSqrowf = endSqcol, endSqrow
    else :
        if board[endSqrow][endSqcol][0] != "b":
            endSqcolf, endSqrowf = endSqcol, endSqrow
            startSqcolf, startSqrowf = startSqcol, startSqrow
        else :
            endSqcolf, endSqrowf = startSqcol, startSqrow
            startSqcolf, startSqrowf = endSqcol, endSqrow

    cv2.imwrite(filename='Initial_Image.jpg', img=roi)
    return startSqrowf, startSqcolf, endSqrowf, endSqcolf


CapturingBoard()
x, y, w, h = CroppingBoard()
finalList2 = getChessSquares()

img = cv2.imread("Initial_Image.jpg")
contours = []
for item in finalList2:
    c = item[0]
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(img, item[1], (cX, cY), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
cv2.imshow("Contours and Squares", img)
cv2.waitKey(0)

# while(True):
#     img = ImageGrab.grab(bbox=(0,0,1000,1000)) #bbox specifies specific region (bbox= x,y,width,height)
#     img_np = np.array(img)
#     frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

#     roi = frame[y: y+h, x: x+w]

#     cv2.imshow("The Board", roi)
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         cv2.imwrite(filename='Image1.jpg', img=roi)

#     elif cv2.waitKey(1) & 0xFF == ord('d'):
#         cv2.imwrite(filename='Image2.jpg', img=roi)

#     elif cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break