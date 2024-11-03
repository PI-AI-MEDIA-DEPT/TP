from cgi import test
from ctypes import *
import sys
import cv2
import numpy as np
from requests.models import ContentDecodingError, Response
import time
import threading, queue
import cv2
import numpy as np
import time
import os

from ultralytics import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import multiprocessing
import requests
import json
#rtspt://admin:people1!@192.168.33.24

class Localization(threading.Thread):
    def __init__(self, url, camid, cs_host, section_name = 'work',model_name="best.pt"):
        threading.Thread.__init__(self,daemon=True)
        print(url, camid)
        self.url = '''rtspsrc location={} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'''.format(url)
        #self.url = 'filesrc location={} ! decodebin ! videoconvert ! appsink'.format(url)
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)
        self.video_start = False
        self.frame_result = np.zeros((1080,1920,3), dtype=np.uint8)
        self.frame_ori = np.zeros((1080,1920,3), dtype=np.uint8)
        self.capture_alive = False
        self.camid = str(camid)
        self.cs_host = cs_host
        self.cs_retrun_camera_info = f"http://{self.cs_host}/api/area/returnCameraAreaInfo/"
        self.section_name = section_name
        self.area_datas = []
        self.alarm_area = []
        self.polygons = []
        self.alpha = 0.4

        self.model0 = YOLO(model_name)
        # self.model1 = YOLO('best.pt')
        # # self.model2 = YOLO('best.pt')
        # # self.model3 = YOLO('best.pt')
        # # self.model4 = YOLO('best.pt')
        # # self.model2 = YOLO('yolov8n.pt')
        self.QR = cv2.QRCodeDetector()

    def run(self):
        self.camPreview(self.url)

    def hex_to_rgb(self, value): 
        value = value.lstrip('#')
        lv = len(value)
        rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        bgr = (rgb[2],rgb[1],rgb[0])
        return bgr

    def get_area_data(self):
        self.area_datas = []
        self.alarm_area = []
        self.polygons = []

        payload = json.dumps({
            "section_name": self.section_name,
            "cam_id": self.camid
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", self.cs_retrun_camera_info, headers=headers, data=payload)
        print("GET : AreaInfo ..",response.text)
        ret = response.json()["response"]

        for r in ret:
            pts = json.loads(r["points"])["points"],
            x = []
            y = []

            for p in pts:
                for i,j in enumerate(p):
                    if i % 2 == 0:
                        j = int(j * 2)
                        x.append(j)
                    else:
                        j = int(j * 2)
                        y.append(j)
            points = []
            for x1,y1 in zip(x,y):
                points.append((x1,y1))

            data = {
                'id': r["id"], 
                'areaName': r["areaName"],
                'areaColor': self.hex_to_rgb(r["areaColor"]),
                'points': points,
                'sendDetectSignal': r["sendDetectSignal"],
                "area" : ""
            }

            self.area_datas.append(data)
        
        for a in  self.area_datas:
            sendDetectSignal = a['sendDetectSignal']
            if sendDetectSignal:
                self.alarm_area.append(a['areaName'])

            pts = [a["points"][0], a["points"][1],a["points"][3],a["points"][2],a["points"][0]]
            poly = {"areaName": a['areaName'], "polygon":Polygon(pts), "areaColor": a["areaColor"]}
            self.polygons.append(poly)

    def inner_area(self, x,y):
        inner = False
        alert = False

        for p in self.polygons:
            point = Point(x, y)
            inner = p['polygon'].contains(point)
            areaName = p['areaName']
            color = p['areaColor']

            if inner :
                if areaName in self.alarm_area:
                    alert = True
                    return inner, areaName ,alert, color
                else:
                    return inner, areaName ,alert, color

          
        return inner, "CommonArea", alert, (0,0,0)
    
    def camPreview(self,url):
        while True:
            if self.video_start:
                break
            else:
                time.sleep(0.06)

        self.get_area_data()
        while True:
            try:
                rval, frame = self.cap.retrieve()
                if rval != self.capture_alive:
                    self.capture_alive = rval
                
                if rval == False:
                    time.sleep(1)
                    #self.new_capture()
                    continue

                frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
                frame_ori = frame.copy()
                overlay = frame.copy()
                try:
                    qr_data, box, straight_qrcode = self.QR.detectAndDecode(frame)
                except Exception as E:
                    print(E)
                results = self.model0.predict(frame, conf=0.4, verbose=False)
                # if self.camid == 0 or self.camid == 2:
                #     results = self.model0.predict(frame, conf=0.4, verbose=False)
                # else:
                #     results = self.model1.predict(frame, conf=0.4, verbose=False)
                # # elif self.camid == 2:area_datas
                #     results = self.model2.predict(frame, conf=0.4, verbose=False)
                # elif self.camid == 3:
                #     results = self.model3.predict(frame, conf=0.4, verbose=False)

                for a in  self.area_datas:
                    pts = [a["points"][0], a["points"][1],a["points"][3],a["points"][2],a["points"][0]]
                    pt = np.array(pts, np.int32)
                    overlay = cv2.fillPoly(overlay,[pt],a["areaColor"])
                    overlay = cv2.polylines(overlay,[pt],False,(0,0,255),2)

                frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)
                
                for ix, r in enumerate(results):
                    boxes = r.boxes
                    for box in boxes:
                        # if box.cls == 0:
                        box_xyxy = box.xyxy[0].cpu().detach().numpy()
                        body_x1  = int(box_xyxy[0])
                        body_y1 = int(box_xyxy[1])            
                        body_x2  = int(box_xyxy[2])
                        body_y2 = int(box_xyxy[3])
                        x1 = int(body_x1 + (body_x2 - body_x1)/ 2)
                        y1 = int(body_y2)
                        #cv2.rectangle(frame, (body_x1,body_y1), (body_x2,body_y2), (0,0,255), 1)

                        # if x1 <= 0 or y1 <= 0:
                        #     continue
                        try:
                            inner, areaname, alert, color = self.inner_area(x1,y1)
                            if inner and areaname in self.alarm_area:
                                cv2.rectangle(frame, (body_x1,body_y1), (body_x2,body_y2), (0,0,255), 1)
                            else:
                                cv2.rectangle(frame, (body_x1,body_y1), (body_x2,body_y2), (255,255,255), 1)
                        except Exception as E:
                            print(E)
                try:
                    if qr_data:
                        cv2.putText(frame, 'QR Data: {}'.format(qr_data), (30,30), 1, 1, (0,0,255),2)
                except Exception as E:
                    print(E)
                self.frame_result = frame
                self.frame_ori = frame_ori
                
            except Exception as E:
                print(E)
                pass 
            #self.capture_alive = False
    def new_capture(self):
        #del self.cap
        time.sleep(2)
        self.cap =  cv2.VideoCapture(self.url)

    def set_video_pos(self, pos):
        self.cap.set(1, pos)

    def grab_frame(self):
        self.cap.grab()
     
class Detect():
    def __init__(self):
        self.cap_alive = {}

    def run(self):   
        container = []
        rtsp1 = {"rtsp": "rtsp://192.168.33.40:8554/mystream", "cam_id":0, "model":"best.pt"}
        rtsp2 = {"rtsp": "rtsp://192.168.33.249:8554/mystream", "cam_id":1, "model":"best.pt"}
        
        # rtsp1 = {"rtsp": "rtsp://192.168.33.12:8554/mystream", "cam_id":0}
        # rtsp2 = {"rtsp": "rtsp://192.168.33.48:8554/mystream", "cam_id":1}
        # rtsp3 = {"rtsp": "rtsp://192.168.33.16:8554/mystream", "cam_id":2}
        # rtsp4 = {"rtsp": "rtsp://192.168.33.17:8554/mystream", "cam_id":3}
        recode_out = cv2.VideoWriter('appsrc ! videoconvert' + \
                    ' ! x264enc speed-preset=ultrafast bitrate=6400 key-int-max=' + str(15) + \
                    ' ! video/x-h264,profile=baseline' + \
                    ' ! rtspclientsink location={}'.format("rtsp://127.0.0.1:8554/front"),
                    cv2.CAP_GSTREAMER, 0, 15, (1920,540), True)
        
        container.append(rtsp1)
        container.append(rtsp2)
        net1 = 0
        net2 = 0
        nets = [net1, net2]
        self.cap_alive = {}
        typeCheckDelay = 60
        cs_host="127.0.0.1:80"

        for index, ct in enumerate(container):
            nets[index] = Localization(url=ct['rtsp'],camid=ct['cam_id'],
                                       cs_host=cs_host,
                                       section_name = 'TP',
                                       model_name = ct['model'])
        time.sleep(1)  

        for n in nets:
            n.start()
            time.sleep(1)

        for n in nets:
            n.video_start = True

        for n in nets:
            self.cap_alive[n.camid] = n.capture_alive 

        # cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("vis", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("vis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.namedWindow("vis", cv2.WND_PROP_FULLSCREEN)
        # cv2.resizeWindow("vis", 1920,1080)

        while True:        
            vis_temp = np.zeros((540,1920, 3), dtype=np.uint8)
            #vis_ori = np.zeros((1080,1920, 3), dtype=np.uint8)

            for n in nets:
                n.grab_frame()

            for index_n, n in enumerate(nets):
                try:
                    vid = n.frame_result
                    ori = n.frame_ori
                    vid = cv2.resize(vid,(960,540))
                    ori = cv2.resize(ori,(960,540))
                except Exception as E:
                    print(E)
                    #n.new_capture()
                    #continue
                
                if index_n == 0:
                    vis_temp[0:540,0:960] = vid

                if index_n == 1:
                    vis_temp[0:540,960:1920] = vid

            #vis_temp = cv2.resize(vis_temp, (1920,1080))

            # cv2.imshow("vis", vis_temp)
            recode_out.write(vis_temp)

            key = cv2.waitKey(1) #60

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        # self.post_log_msg(HOST, industry, msg)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    # time.sleep(5)
    detect = Detect().run()

