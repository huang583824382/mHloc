import socket
import cv2
import struct
import numpy as np
import threading
import json
from typing import Union
import numpy as np
from hloc.my_localize_lib import localize_image, MainWork

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

HOST = '192.168.227.201'
PORT = 5838

def img_rotate(src, angel):
    """逆时针旋转图像任意角度

    Args:
        src (np.array): [原始图像]
        angel (int): [逆时针旋转的角度]

    Returns:
        [array]: [旋转后的图像]
    """
    h,w = src.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angel, 1.0)
    # 调整旋转后的图像长宽
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))
    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2
    # 旋转图像
    rotated_img = cv2.warpAffine(src, M, (rotated_w,rotated_h))

    return rotated_img

class Singleton(object):
    def __init__(self, PackageWorker):
        self._PackageWorker = PackageWorker
        self._instance = {}

    def __call__(self):
        if self._PackageWorker not in self._instance:
            self._instance[self._PackageWorker] = self._PackageWorker()
        return self._instance[self._PackageWorker]


@Singleton
class Package:
    frameHead = 58
    types = {'image800x600': 0, 'text': 1, 'poseInfo': 2, 'errorInfo': 3, '6dof': 4}

    def decodeJSON(self, s: str):
        None
    
    def generateStream(self, type: int, data: Union[bytes, None])->bytes:
        length = 0
        if data is not None:
            length = len(data)
        frame = struct.pack('<hhi{}s'.format(length), self.frameHead, type, length, data)
        return frame

    def parseHeadAndRecvAll(self, frameHead: bytes, conn: socket):
        head = struct.unpack('<hhi', frameHead)
        frameHead = head[0]
        if frameHead != self.frameHead: 
            raise ValueError('Get Error Package')
        type = head[1]
        length = head[2]
        left=length
        data = bytes()
        while True:
            data = data + conn.recv(left)
            dataLen = len(data)
            left=length-dataLen
            if left == 0:
                # recv finish
                break;
        return type, data

    
    def parseStream(self, frame: bytes):
        head = struct.unpack('<hhi', frame[:8])
        frameHead = head[0]
        if frameHead != self.frameHead: 
            raise ValueError('Get Error Package')
        type = head[1]
        print('Get Package, type is', list(self.types.keys())[type])
        length = head[2]
        print('length:', length, 'actual data length =', len(frame)-8)
        payloadObj = self.parsePayload(type, frame[8:])
        return type, payloadObj

    def parsePayload(self, type: int, payload: bytes):
        if type == self.types['text']:
            print('Get text message:', payload.decode())
            return payload.decode()
        elif type == self.types['image800x600']:
            print('Get image message')
            d = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(d, cv2.IMREAD_COLOR)
            img = img_rotate(img, 90)
            cv2.imwrite('tmp.jpg', img) 
            # cv2.imshow('img', img)
            # cv2.waitKey()
            global semQuery
            semQuery.release()
            return img
        
def queryMain():
    print("queryMain in")
    # create Mainwork
    mainWork = MainWork()
    # loop
    global semQuery, semSend, send_buffer
    while True:
        semQuery.acquire()
        # do query and put result in some variable, 
        ret, _, log , res_fig = localize_image('datasets/My_lib_dataset/mapping4', '', 'tmp.jpg', overwrite=True)
        print("query success")
        # preprocess data
        ret['qvec'] = ret['qvec'].tolist()
        ret['tvec'] = ret['tvec'].tolist()
        ret['camera']['params'] = ret['camera']['params'].tolist()
        log['keypoints_query'] = log['keypoints_query'].tolist()

        with open('json.txt', 'w') as fp:
            json.dump(log, fp, cls=NpEncoder)
            fp.close()

        # send the result
        semSend.release()

def sendMain(conn: socket):
    print("sendMain start")
    packageWorker = Package()
    while True:
        semSend.acquire()
        # send the data in buffer
        with open('json.txt', 'r') as fp:
            json = fp.read()
            fp.close()
        print("in send", json)
        replyFrame = packageWorker.generateStream(packageWorker.types['poseInfo'], json.encode())
        conn.send(replyFrame)
        print("send success")


def serverMain():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(5)
    packageWorker = Package()
    with open("json.txt", 'r') as f:
        jsonstr = f.read();
    i=0
    print ('load jsonstr: ', len(jsonstr))

    print ('Server start at: %s:%s' %(HOST, PORT))
    print ('wait for connection...')
    while True:
        conn, addr = s.accept()
        print( 'Connected by ', addr)
        # global lockRes
        # create a send thread
        t_send = threading.Thread(target=sendMain, args=(conn,))
        t_send.start()
        # listening
        while True:
            packageHead = conn.recv(8)
            type, data = packageWorker.parseHeadAndRecvAll(packageHead, conn)
            packageWorker.parsePayload(type, data)
                
            # replyFrame = packageWorker.generateStream(packageWorker.types['poseInfo'], jsonstr.encode())
            # conn.send(replyFrame)
            # print('reply success:', i)
            # i=i+1

            # type, payload = packageWorker.parseStream(data)

send_buffer = None
res_valid = False

semQuery = threading.Semaphore(0)
semSend = threading.Semaphore(0)
# lockRes = threading.Lock()

t_server = threading.Thread(target=serverMain)
t_query = threading.Thread(target=queryMain)

t_query.start()
t_server.start()
t_server.join()
t_query.join()