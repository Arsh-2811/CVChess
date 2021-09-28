import cv2

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap = cv2.flip(cap, 1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

r, frame = cap.read()
...
print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))