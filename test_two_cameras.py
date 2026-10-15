
import cv2
import time

PHONE_URL = "http://10.213.204.242:8080/video"

webcam = cv2.VideoCapture(0)
phone = cv2.VideoCapture(PHONE_URL)

print("Laptop webcam connected:", webcam.isOpened())
print("Mobile camera connected:", phone.isOpened())

try:
    while True:
        ok1, frame1 = webcam.read()
        ok2, frame2 = phone.read()

        if ok1:
            cv2.imshow("SAFE-X - Laptop Webcam", frame1)

        if ok2:
            cv2.imshow("SAFE-X - Mobile Camera", frame2)

        if not ok1 and not ok2:
            print("No camera frames received. Check camera connections.")
            time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    webcam.release()
    phone.release()
    cv2.destroyAllWindows()