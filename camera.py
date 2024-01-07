import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QFrame, QHBoxLayout, QVBoxLayout,
                             QAction, QMenuBar, QFileDialog, QMessageBox)
from numpy import ndarray, array



import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import random
import torch
# from distyolov7
from utils.plots import plot_one_box

from signal import signal, SIGPIPE, SIG_DFL  
signal(SIGPIPE,SIG_DFL)



from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs
import threading
import json

# 定义处理 POST 请求的类
class PostHandler(BaseHTTPRequestHandler):
    # 处理 POST 请求
    def do_POST(self):
        global received_data
        # 获取 POST 数据
        content_length = int(self.headers.get('Content-Length'))
        post_data = self.rfile.read(content_length).decode('utf-8')
        parsed_data = parse_qs(post_data)
        # 在这里处理接收到的 POST 数据，这里只是简单打印出来
        print(f'Received POST data: {parsed_data}')
        for key in parsed_data:
            received_data = key
            received_data = received_data[:-1]
        # 设置响应状态码
        # self.send_response(200)
        # self.end_headers()
        # self.wfile.write(b'POST request received successfully')

# 定义处理 GET 请求的类
class GetHandler(BaseHTTPRequestHandler):
    # 处理 GET 请求
    def do_GET(self):
        # 获取接收到的 POST 数据（这里假设存储在一个全局变量中）
        global received_data
        # received_data = "137,10-144,21-"
        
        if received_data:
            # 设置响应状态码
            self.send_response(200)

            # 设置响应头
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            # 将接收到的数据转换为 JSON 格式
            response_data = {'message': received_data}
            json_data = json.dumps(response_data)
            print(f'send data = {json_data}')
            # 发送 JSON 数据作为响应内容
            self.wfile.write(json_data.encode('utf-8'))
        else:
            # 设置响应状态码
            # self.send_response(404)
            # self.end_headers()
            # self.wfile.write(b'No POST data available')
            response_data = {'message': '-'}
            json_data = json.dumps(response_data)
            print(f'send data = {json_data}')
            # 发送 JSON 数据作为响应内容
            self.wfile.write(json_data.encode('utf-8'))

# 定义多线程的 HTTP 服务器类
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

# 启动服务器
def run(server_class, handler_class, post_port, get_port):
    # 创建 POST 服务器
    # post_server_address = ('192.168.2.101', post_port)
    # post_httpd = server_class(post_server_address, PostHandler)
    # print(f'Starting POST server on port {post_port}...')
    
    # 创建 GET 服务器
    get_server_address = ('192.168.137.180', get_port)
    get_httpd = server_class(get_server_address, GetHandler)
    print(f'Starting GET server on port {get_port}...')
    
    # 启动两个服务器
    # post_server_thread = threading.Thread(target=post_httpd.serve_forever)
    get_server_thread = threading.Thread(target=get_httpd.serve_forever)


    # post_server_thread.start()
    get_server_thread.start()





class trt_model():
    def __init__(self):
        self.load_model()

    def load_model(self):
        global context, stream, host_in, host_out, devide_in, devide_out
        print('Load model')
        # Load trt model
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open('./tmp/best_js.trt', 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Prepare TRT execution context, CUDA stream and necessary buffer
        context = self.engine.create_execution_context()
        stream = cuda.Stream()
        host_in = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        host_out = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        devide_in = cuda.mem_alloc(host_in.nbytes)
        devide_out = cuda.mem_alloc(host_out.nbytes)

        self.names = [ 'pedestrian', 'car', 'motorcycle', 'truck']
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def inference(self, img):
        global context, stream, host_in, host_out, devide_in, devide_out
        # image preprocessing
        img0 = letterbox(img, stride=32)[0]
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.array(img, dtype=np.float32) # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # img = torch.from_numpy(img).to('cpu')
        # Inferece
        bindings = [int(devide_in), int(devide_out)]
        np.copyto(host_in, img.ravel())
        cuda.memcpy_htod_async(devide_in, host_in, stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_out, devide_out, stream)
        stream.synchronize()

        # reshape and remove duplicate
        pred = np.unique(host_out.reshape(self.engine.get_binding_shape(1)), axis=0)
        
        # plot result
        self.plot(img0, pred)
        
        return img0, pred

    def plot(self, img, pred):
        global received_data
        # Rescale boxes from img_size to img0 size
        det = torch.from_numpy(pred).float()
        # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        received_data = ""
        # Print result on image and save
        for *xyxy, conf, dist, cls in reversed(det):
            if xyxy[0] == 0 and dist == 0:
                continue
            received_data += str(int((xyxy[2] + xyxy[0])/2))
            received_data += ","
            received_data += str(int(dist*100))
            received_data += "-"
            label = f'{self.names[int(cls)]} conf:{conf:.2f} dist:{dist:.2f}m'
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)], line_thickness=1)
        if received_data == "":
            received_data = "-"
        print(f'received_data = {received_data}')




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class CameraWorkerThread(QThread):
    """Worker Thread for capture camera"""
    frame_data_updated = pyqtSignal(ndarray)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.model = trt_model()

    def run(self):
        global model
        self.capture =  cv2.VideoCapture(0)

        while self.parent.thread_is_running:
            
            ret_val, frame = self.capture.read()
            if not ret_val:
                print("cannot capture camera")
                break
            # resize img
            t1 = time.time()
            frame = self.model.inference(frame)
            t2 = time.time()
            cv2.putText(frame, f'fps : {1000/(t2-t1)}', (0,0), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2, cv2.LINE_AA)
            
            #
            frame = frame.transpose(1, 2, 0).copy()

            self.frame_data_updated.emit(frame)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initializeUI()
        self.thread_is_running = False

    def initializeUI(self):
        ''' Initialize Main window'''
        self.setMinimumSize(800, 500)
        self.setWindowTitle('distyolo camera')
    
        self.setupWindow()

    def showEvent(self, event):
        self.startCamera()
        event.accept()

    def setupWindow(self):
        ''' setup widgets in the Main window'''
        self.camera_display_label = QLabel()
        self.camera_display_label.setObjectName('CameraLabel')
        
        main = QVBoxLayout()
        main.addWidget(self.camera_display_label)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

    def startCamera(self):
        self.thread_is_running = True
        
        # camera worker thread
        self.camera_thread_worker = CameraWorkerThread(self)

        # Connect to the thread;s signal to update the frames in the video_display_label
        self.camera_thread_worker.frame_data_updated.connect(self.updateCameraFrames)
        self.camera_thread_worker.start()

    def stopCamera(self):
        if self.thread_is_running == True:
            self.thread_is_running = False
            self.camera_thread_worker.stopThread()

            self.camera_display_label.clear()
            
    def updateCameraFrames(self, frame):
        self.frame = frame
        height, width, channels = frame.shape
        bytes_per_line = width * channels
        Qt_image = QImage(frame, width, height, bytes_per_line, QImage.Format_RGB888)

        self.camera_display_label.setPixmap(QPixmap.fromImage(Qt_image).scaled(self.camera_display_label.width(), self.camera_display_label.height(), Qt.KeepAspectRatioByExpanding))
        self.camera_display_label.setScaledContents(True)

    def closeEvent(self, event):
        if self.thread_is_running == True:
            self.camera_thread_worker.quit()

if __name__ == '__main__':
    if 0:
        model = trt_model()
        img = cv2.imread('./nuscenes/new.jpg')
        img = cv2.resize(img, (640, 480))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img, dtype=np.float32).transpose(2, 0, 1) # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        model.inference(img)

    # load model and camera
    capture =  cv2.VideoCapture(0)
    model = trt_model()

    received_data = ''  # 存储接收到的 POST 数据
    
    run(ThreadedHTTPServer, PostHandler, 8000, 8001)

    print("start capture and inference")
    while 1:
        ret_val, frame = capture.read()
        t1 = time.time()
        frame, pred = model.inference(frame)
        t2 = time.time()
        cv2.putText(frame, f'fps : {1/(t2-t1)}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
        # print( f'fps : {1/(t2-t1)}')
        frame = cv2.resize(frame,(320, 240))
        cv2.imwrite('./camera_out.jpg', frame)
        # cv2.imshow('live', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    capture.realease()
    cv2.destroyAllWindows()
    exit(0)
    import socket
    HOST = "0.0.0.0"
    PORT = 7001
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        while 1:
            print("wait connection")
            conn, addr = s.accept()
            
            try:
                with conn:
                    while 1:  
                        ret_val, frame = capture.read()
                        if not ret_val:
                            print("cannot capture camera")
                            break
                        # resize img
                        t1 = time.time()
                        frame, pred = model.inference(frame)
                        t2 = time.time()
                        cv2.putText(frame, f'fps : {1/(t2-t1)}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                        print( f'fps : {1/(t2-t1)}')
                        
                        msg = []
                        n = len(pred)
                        for p in pred:
                            # xyxy2xywh
                            x = int((p[0] + p[2])/2)/640
                            y = int((p[1] + p[3])/2)/480
                            w = int((p[2] - p[0])/2)/640
                            h = int((p[3] - p[1])/2)/480
                            if w==0 and h==0:
                                n-=1
                                continue
                            msg.append(f'{p[0]/640} {p[1]/480} {p[2]/640} {p[3]/480} {p[5]}\r')
                        conn.send(f'{len(msg)}\r'.encode())
                        time.sleep(0.001)
                        for m in msg:
                            conn.send(m.encode())
                            time.sleep(0.001)
                        time.sleep(0.001)
                        conn.send('finish\r'.encode())
                    
                        #frame = frame.transpose(1, 2, 0).copy()
                        cv2.imwrite('./camera_out.jpg', frame)
            except KeyboardInterrupt:
                print("keyboadr interrupt")
                
            finally:
                print("finally")

    '''
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    '''
