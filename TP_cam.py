from cgi import test
from ctypes import *
import sys
import cv2
import numpy as np
import time
import threading, queue
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import requests
import json
import argparse
#rtspt://admin:people1!@192.168.33.24

class Localization(threading.Thread):
    def __init__(self, url, camid, cs_host, section_name = 'work',model_name="best.pt",rtsp_out = "TP1"):
        threading.Thread.__init__(self,daemon=True)
        print(url, camid, rtsp_out)
        self.url = url
        #self.url = '''rtspsrc location={} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'''.format(url)
        # self.cap = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)

        # self.url = 'filesrc location={} ! decodebin ! videoconvert ! appsink'.format(url)
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
         
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
        self.QR = cv2.QRCodeDetector()

        self.recode_out = cv2.VideoWriter('appsrc ! videoconvert' + \
            ' ! x264enc speed-preset=ultrafast bitrate=6400 key-int-max=' + str(15) + \
            ' ! video/x-h264,profile=baseline' + \
            ' ! rtspclientsink location={}'.format("rtsp://127.0.0.1:8554/{}".format(rtsp_out)),
            cv2.CAP_GSTREAMER, 0, 15, (960,540), True)

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
        self.get_area_data()
        while True:
            try:
                rval, frame = self.cap.read()
                if rval == False:
                    time.sleep(1)
                    #self.set_video_pos()
                    self.new_capture()
                    continue

                frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
                frame_ori = frame.copy()
                overlay = frame.copy()
                try:
                    qr_data, box, straight_qrcode = self.QR.detectAndDecode(frame)
                except Exception as E:
                    print(E)
                results = self.model0.predict(frame, conf=0.4, verbose=False)

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
                # self.frame_ori = frame_ori
                self.recode_out.write(frame)
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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    # rtsp1 = {"rtsp": "drive1.avi", "cam_id":0, "model":"best.pt"}
    # rtsp2 = {"rtsp": "drive2.avi", "cam_id":1, "model":"best2.pt"}
    # rtsp3 = {"rtsp": "drive3.avi", "cam_id":2, "model":"best3.pt"}
    # rtsp4 = {"rtsp": "drive4.avi", "cam_id":3, "model":"last.pt"}
    parser = argparse.ArgumentParser(description="Process camera input and output configurations.")
    # Define arguments as optional with default values
    parser.add_argument("--src", type=str, default="drive1.avi", help="input source file")
    parser.add_argument("--camid", type=int, default=0, help="camera ID")
    parser.add_argument("--out", type=str, default="TP1", help="rtsp output path")
    parser.add_argument("--section", type=str, default="TP", help="section name")
    parser.add_argument("--model", type=str, default="best.pt", help="model name")

    args = parser.parse_args()

    cs_host = "127.0.0.1"
    loc = Localization(url=args.src, camid=args.camid,
                    cs_host=cs_host,
                    section_name=args.section,
                    rtsp_out=args.out,
                    model_name=args.model)
    
    loc.start()
    loc.join()

