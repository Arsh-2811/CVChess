import cv2

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    try:
        check, frame = webcam.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='LOL.jpg', img=frame)
            # webcam.release()
            # break
        if key == ord('d'):
            cv2.imwrite(filename='LMAO.jpg', img=frame)
            # webcam.release()
            # break
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