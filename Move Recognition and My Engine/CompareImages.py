from skimage.metrics import structural_similarity as ssim
import imutils
import cv2

imageA = cv2.imread("LOL.jpg")
imageB = cv2.imread("LMAO.jpg")

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts1, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = []
for cnt in cnts1:
	area = cv2.contourArea(cnt)
	if area >= 600 and area <= 1400 :
		cnts.append(cnt)
	                    # 600 - 1400
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cv2.drawContours(imageA, cnts, -1, (255, 0, 0))

cnts.sort (key = lambda x: cv2.contourArea (x), reverse = True)
max_cnt = cnts[0]
x, y, w, h = cv2.boundingRect(max_cnt)

max_cnt2 = cnts[1]
a, b, c, d = cv2.boundingRect(max_cnt2)

cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.rectangle(imageA, (a, b), (a + c, b + d), (0, 0, 255), 2)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)