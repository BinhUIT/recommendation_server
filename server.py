from http.server import BaseHTTPRequestHandler
import os,re
import time
from deepface import DeepFace
from http.server import HTTPServer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import timm
import json
import cv2
import mediapipe as mp
import math
from sklearn.metrics import accuracy_score
HOST_NAME='localhost'
port=8082
UPLOAD_DIR='IMG'
TRIANGLE_SHAPE_RAITO=2
APPLE_SHAPE_RAITO=0.8
class detect_data:
     def __init__(self):
          self.age=0
          self.gender=False
          self.shape="Undefined"
          self.body_shape="Undefined"
     def set_age(self,age):
          self.age=age
     def set_shape(self,shape):
         self.shape=shape
     def set_body_shape(self,body_shape):
         self.body_shape=body_shape
     def set_gender(self, gender):
          self.gender=gender
     def to_dict(self):
        return {
            "age": self.age,
            "gender": self.gender,
            "shape":self.shape,
            "body_shape":self.body_shape
        }
class GenderDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["gender"], dtype=torch.long)  # 0 hoặc 1
        return img, label
def calc_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def body_shape_calc(left_shoulder, right_shoulder,left_hip,right_hip):
  shoulder_distance = calc_distance(left_shoulder,right_shoulder)
  hip_distance = calc_distance(left_hip,right_hip)
  raito=shoulder_distance/hip_distance
  if(raito>TRIANGLE_SHAPE_RAITO):
    return "Triangle"
  if(raito<APPLE_SHAPE_RAITO):
    return "Apple"
  return "Rectangle"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])          
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)


model.load_state_dict(torch.load("gender_effb3.pth", map_location="cpu"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)
class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200,"Success")
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            message = "<h1>Hello from Python server!</h1>"
            self.wfile.write(message.encode('utf-8'))
    def do_POST(self):
         detect= detect_data()
         if self.path == '/test':
            print("---- POST RECEIVED ----")
            print("Path:", self.path)
            print("Headers:", self.headers)
            print("Content-Length:", self.headers.get('Content-Length'))
            content_type = self.headers.get('Content-Type')
            content_length = int(self.headers.get('Content-Length', 0))
            match = re.match(r'multipart/form-data;\s*boundary="?([^";]+)"?', content_type)
            if not match:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid Content-Type"}')
                return
            boundary = match.group(1).encode()
            body = self.rfile.read(content_length)
            parts = body.split(b'--' + boundary)

            for part in parts:
                if b'Content-Disposition' in part and b'name="file"' in part:
                        
                    match = re.search(br'filename="([^"]+)"', part)
                    if not match:
                            continue
                    filename = match.group(1).decode()
                        
                    file_data_start = part.find(b'\r\n\r\n') + 4
                    file_data = part[file_data_start:]
                    file_data = file_data.rstrip(b'\r\n--')

                        # Ghi file
                    save_path = os.path.join(UPLOAD_DIR, filename)
                    with open(save_path, 'wb') as f:
                        f.write(file_data)
                    result = DeepFace.analyze(img_path=save_path, actions=['age'])
                    detect.set_age(result[0]["age"])
                    img = Image.open(save_path).convert("RGB")



                    
                    img_tensor = transform(img).unsqueeze(0)  

                    
                    with torch.no_grad():
                        preds = model(img_tensor)
                        pred_class = preds.argmax(1).item()
                    detect.set_gender(pred_class)
                    img = cv2.imread(save_path)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)
                    if result.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            img,
                            result.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                        )

                        landmarks = []
                        h, w, c = img.shape
                        for id, lm in enumerate(result.pose_landmarks.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            landmarks.append((cx, cy))
                            print(f"ID {id}: x={cx}, y={cy}, z={lm.z:.4f}")


                        head = landmarks[0]                # NOSE
                        left_shoulder = landmarks[11]
                        right_shoulder = landmarks[12]
                        left_hip = landmarks[23]
                        right_hip = landmarks[24]
                        left_ankle = landmarks[27]
                        right_ankle = landmarks[28]


                        shoulder_center = ((left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2)
                        hip_center = ((left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2)
                        feet_center = ((left_ankle[0]+right_ankle[0])/2, (left_ankle[1]+right_ankle[1])/2)


                        height_ratio = calc_distance(head, feet_center)


                        shoulder_width = calc_distance(left_shoulder, right_shoulder)
                        hip_width = calc_distance(left_hip, right_hip)
                        body_width = (shoulder_width + hip_width) / 2

                        shape_ratio = body_width / height_ratio


                        if shape_ratio < 0.20:
                            shape = "thin"
                            detect.set_shape(shape)
                        elif shape_ratio < 0.27:
                            shape = "normal"
                            detect.set_shape(shape)
                        else:
                            shape = "fat"
                            detect.set_shape(shape)
                        print(f"Analysis")
                        print(f"Height: {height_ratio:.2f} px")
                        print(f"Body with: {body_width:.2f} px")
                        print(f"Shape raito: {shape_ratio:.3f} → {shape}")
                        body=body_shape_calc(left_shoulder, right_shoulder,left_hip,right_hip)
                        print(f"Body shape: {body}")
                        detect.set_body_shape(body)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(detect.to_dict()).encode())
                        

     
httpd = HTTPServer((HOST_NAME,port),Server)
print("Start server")
try:
    httpd.serve_forever()
except KeyboardInterrupt:
        pass
httpd.server_close()
