import socket
import cv2
import struct
import numpy as np
import threading
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from hloc.my_localize_lib import localize_image, MainWork

HOST = '172.22.9.10'
PORT = 5838


class MyDash(threading.Thread):
    app = dash.Dash()
    fig_old = None
    def __init__(self, thread_name):
        super(MyDash, self).__init__(name=thread_name)

    def run(self):
        # print('initing...')
        # global app, fig
        self.app.layout = html.Div([dcc.Graph(id='live_figure'), dcc.Interval(id="interval", interval=1*1000 ,n_intervals=0)])
        @self.app.callback(Output('live_figure', 'figure'), [Input('interval', 'n_intervals')])
        def update_data(n_intervals):
            # print('in update_data', n_intervals)
            global fig
            # print('in update')
            # if self.fig_old is None:
            #     self.fig_old = fig
            # elif self.fig_old == fig:

            return fig

        self.app.run_server(debug=True, use_reloader=False)

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
    
    def generateStream(self, type: int, data: bytes)->bytes:
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
        self.parsePayload(type, data)
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
            cv2.imwrite('datasets/My_lib_dataset/query/tmp.jpg', img)

            return img

sem = threading.Semaphore(0)
packageWorker = Package()
type = None
fig = None
t_display = None
queryLock = threading.Lock()
mainWork = MainWork()

def solveQuery():
    # hlocMain = MainWork()
    print('solveQuery in')
    # global type, obj, sem, fig, show_fig_sem
    global sem, fig, queryLock
    while True:
        # await event.wait()
        sem.acquire()
        queryLock.acquire()
        print('event trigger success')
        if type == packageWorker.types['image800x600']:
            # print('test locating...')
            ret, _, _ , res_fig = localize_image('datasets/My_lib_dataset/mapping4', 'datasets/My_lib_dataset/query', 'tmp.jpg', overwrite=True)
            if ret['success'] is True:
                fig = res_fig
                print('success!!')
                global t_display
                if t_display is None:
                    t_display = MyDash('Hello')
                    t_display.start()
        queryLock.release()

def serverMain():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(5)

    print ('Server start at: %s:%s' %(HOST, PORT))
    print ('wait for connection...')
    global type, queryLock
    while True:
        conn, addr = s.accept()
        print( 'Connected by ', addr)

        while True:
            packageHead = conn.recv(8)
            type, data = packageWorker.parseHeadAndRecvAll(packageHead, conn)
            if type == packageWorker.types['image800x600']:
                if queryLock.locked():
                    continue
                sem.release()
            # type, payload = packageWorker.parseStream(data)

t_server = threading.Thread(target=serverMain)
t_hloc = threading.Thread(target=solveQuery)
t_hloc.start()
t_server.start()
t_server.join()
t_hloc.join()