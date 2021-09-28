import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import imutils

width = 480
height = 480

def findSquareContours(CalibratingImg):
    chessBoardOriginal = cv2.imread(CalibratingImg)

    chessBoard = chessBoardOriginal.copy()
    chessBoard = cv2.cvtColor(chessBoard, cv2.COLOR_BGR2GRAY)
    chessBoard = cv2.Canny(chessBoard, 50, 150)

    cdst = cv2.cvtColor(chessBoard, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(chessBoard, 1, np.pi / 180, 100, None, 0, 0)

    img = np.zeros(chessBoard.shape, dtype=np.uint8)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    req_cntrs = []

    count = 0
    index = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 900 and area < 3600:
            req_cntrs.append(cnt)
            # cv2.drawContours(chessBoardOriginal, contours,
            #                  index, (0, 255, 0), 2)
            count += 1
        index += 1

    return req_cntrs

def ImageCapture():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(1)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='Images/CalibratingImage.jpg', img=frame)
                webcam.release()
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

def CroppingImage(image):

    img = cv2.imread(image)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 162])
    mask = cv2.inRange(imgHSV, lower, upper)
    mask = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mx = (0, 0, 0, 0)
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx

    roi = img[y + 15:y+h - 15, x + 15:x+w - 15]
    cv2.imwrite("Images/Image_crop.jpg", roi)

    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 2)
    cv2.imwrite('Images/Image_cont.jpg', img)

    return x, y, w, h

def cntrs2squares():

    cxf, cyf = 0, 0
    cxs, cys = 0, 0
    cxt, cyt = 0, 0

    sqCntrs = findSquareContours("Images/Image_crop.jpg")
    counter = 0
    for c in sqCntrs:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if counter == 7 :
            cxf, cyf = cX, cY
        if counter == 6 :
            cxs, cys = cX, cY
        if counter == 15 :
            cxt, cyt = cX, cY
        counter += 1
    
    finalList = []

    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8"]

    length = cxs - cxf
    height = cyf - cyt
    for i in range(8):
        for j in range(8):
            new_point = (cxf + i*length, cyf - j*height)
            for cnt in sqCntrs :
                if cv2.pointPolygonTest(cnt, new_point, False) == 1.0 :
                    chessSquare = letters[i] + numbers[j]
                    finalList.append([cnt, chessSquare])

    return finalList

def pieceMovement():
    imageA = cv2.imread("LOL.jpg")
    imageB = cv2.imread("LMAO.jpg")

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts.sort (key = lambda x: cv2.contourArea (x), reverse = True)
    max_cnt = cnts[0]
    x, y, w, h = cv2.boundingRect(max_cnt)

    max_cnt2 = cnts[1]
    a, b, c, d = cv2.boundingRect(max_cnt2)
    return(x, y, w, h), (a, b, c, d)

'''
The main code starts from here
'''
ImageCapture()
x, y, w, h = CroppingImage("Images/CalibratingImage.jpg")
finalList = cntrs2squares()

'''Initializing the Board'''
if len(finalList) == 64:
    # board = []
    # for c in range(8):
    #     lst = []
    #     for r in range(8):
    #         if c == 1 :
    #             piece = "wP"
    #         elif c == 6 :
    #             piece = "bP"
    #         elif c == 0 :
    #             if r == 0 :
    #                 piece = "wR"
    #             if r == 1 :
    #                 piece = "wN"
    #             if r == 2 :
    #                 piece = "wB"
    #             if r == 3 :
    #                 piece = "wQ"
    #             if r == 4 :
    #                 piece = "wK"
    #             if r == 5 :
    #                 piece = "wB"
    #             if r == 6 :
    #                 piece = "wN"
    #             if r == 7 :
    #                 piece = "wR"
    #         elif c == 7 :
    #             if r == 0 :
    #                 piece = "bR"
    #             if r == 1 :
    #                 piece = "bN"
    #             if r == 2 :
    #                 piece = "bB"
    #             if r == 3 :
    #                 piece = "bQ"
    #             if r == 4 :
    #                 piece = "bK"
    #             if r == 5 :
    #                 piece = "bB"
    #             if r == 6 :
    #                 piece = "bN"
    #             if r == 7 :
    #                 piece = "bR"
    #         else :
    #             piece = "--"
    #         print(piece)
    #         lst.append([finalList[8*r + c][0], finalList[8*r + c][1], piece])
    #     board.append(lst)

    # for r in range(8):
    #     for c in range(8):
    #         print(board[r][c][2], end="  ")
    #     print("\n")
    pass
    # print(finalList)
else :
    print("Scanning Error!!")

webcam = cv2.VideoCapture(1)
whiteTomove = True

while True:
    check, chessBoardOriginal = webcam.read()
    chessBoardOriginal = chessBoardOriginal[y + 15:y+h - 15, x + 15:x+w - 15]

    # for item in finalList:
    #     c = item[0]
    #     M = cv2.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     cv2.putText(chessBoardOriginal, item[1],
    #                 (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    cv2.imshow("Live Feed", chessBoardOriginal)

    key = cv2. waitKey(1)
    if key == ord('s'):
        cv2.imwrite(filename='LOL.jpg', img=chessBoardOriginal)
    if key == ord('d'):
        cv2.imwrite(filename='LMAO.jpg', img=chessBoardOriginal)
    if key == ord('f'):
        end, start = pieceMovement()
        p, q, r, s = end
        a, b, c, d = start
        endPnt = p+r//2, q+s//2
        startPnt = a+c//2, b+d//2

        for item in finalList:
            cnt = item[0]
            if cv2.pointPolygonTest(cnt, endPnt, False) == 1.0 :
                endSq = item[1]
            if cv2.pointPolygonTest(cnt, startPnt, False) == 1.0 :
                startSq = item[1]
        print(startSq + endSq, whiteTomove)
        cv2.imwrite(filename='LOL.jpg', img=chessBoardOriginal)
        whiteTomove = not whiteTomove

    elif key == ord('q'):
        webcam.release()
        break