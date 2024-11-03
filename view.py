import cv2
import numpy as np

front_url = "rtsp://127.0.0.1:8554/front"
front_url = '''rtspsrc location={} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'''.format(front_url)
front = cv2.VideoCapture(front_url, cv2.CAP_GSTREAMER)

back_url = "rtsp://127.0.0.1:8554/back"
back_url = '''rtspsrc location={} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'''.format(back_url)
back = cv2.VideoCapture(back_url, cv2.CAP_GSTREAMER)

cv2.namedWindow("vis", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("vis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        

while True:
    vis_temp = np.zeros((1080,1920, 3), dtype=np.uint8)
    _, front_frame = front.read()
    _, back_frame = back.read()

    vis_temp[0:540,0:1920] = front_frame
    vis_temp[540:1080,0:1920] = back_frame


    key = cv2.waitKey(1)

    cv2.imshow("vis", vis_temp)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey(0)

cv2.destroyAllWindows()

