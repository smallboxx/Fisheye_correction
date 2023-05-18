import cv2
import time
import os

cap = cv2.VideoCapture(0)

for i in range(20):
    time.sleep(3)
    ret, frame = cap.read()
    
    if not ret:
        break
    
    cv2.imshow('frame', frame)
    cv2.imwrite(f"img{i}.jpg",frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()