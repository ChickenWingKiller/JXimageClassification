import cv2

cap = cv2.VideoCapture(0)
n = 0
while True:
    ret, frame = cap.read()
    cv2.imshow('Live Video', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.imwrite("C:\\Users\\lyf19\\Desktop\\Image\\" + str(n) + ".jpg", frame)
    #     n += 1
    cv2.waitKey(1)
    # cv2.imwrite("C:\\Users\\lyf19\\Desktop\\Image\\" + str(n) + ".jpg", frame)
cap.release()
cv2.destroyAllWindows()
