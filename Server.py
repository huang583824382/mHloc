import asyncio
import websockets
import struct
import cv2
import numpy as np
from hloc.my_localize_lib import localize_image
import threading
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


class MyDash(threading.Thread):
    app = dash.Dash()
    fig_old = None
    def __init__(self, thread_name):
        super(MyDash, self).__init__(name=thread_name)

    def run(self):
        print('initing...')
        # global app, fig
        self.app.layout = html.Div([dcc.Graph(id='live_figure'), dcc.Interval(id="interval", interval=6*1000 ,n_intervals=0)])

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
    types = {'image800x600': 0, 'text': 1, 'poseInfo': 2, 'errorInfo': 3}
    
    def generateStream(self, type: int, data: bytes)->bytes:
        length = len(data)
        frame = struct.pack('>hhi{}s'.format(length), self.frameHead, type, length, data)
        return frame
    
    def parseStream(self, frame: bytes):
        head = struct.unpack('>hhi', frame[:8])
        frameHead = head[0]
        if frameHead != self.frameHead: 
            raise ValueError('Get Error Package')
            return None
        type = head[1]
        print('Get Package, type is', list(self.types.keys())[type])
        length = head[2]
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
            cv2.imwrite('datasets/My_lib_dataset/query/tmp.jpg', img)
            # cv2.imshow('img', img)
            # cv2.waitKey()

            return img
        
packageWorker = Package()
sem = asyncio.Semaphore(0)
show_fig_sem = asyncio.Semaphore(0)
imgToLoc = None
type = None
obj = None
fig = None
t_display = None

async def echo(websocket, path):
    async for message in websocket:
        global type, obj, sem
        type, obj = packageWorker.parseStream(message)
        if type == packageWorker.types['image800x600']:
            # localize_image('datasets/My_lib_dataset/mapping3', 'datasets/My_lib_dataset/query', 'tmp.jpg')
            # event.set()
            sem.release()
            print('event set success')

        # message = "I got your message: {}".format(message)
        # await websocket.send(message)

# 每次查询图片的特征不应该写入h5文件，不然会越来越大
async def SolveQuery():
    # hlocMain = MainWork()
    print('SolveQuery in')
    global type, obj, sem, fig, show_fig_sem
    while True:
        # await event.wait()
        await sem.acquire()
        print('event trigger success')
        if type == packageWorker.types['image800x600']:
            ret, _, _ , res_fig = localize_image('datasets/My_lib_dataset/mapping3', 'datasets/My_lib_dataset/query', 'tmp.jpg', overwrite=True)
            if ret['success'] is True:
                fig = res_fig
                print('fig setted')
                global t_display
                if t_display is None:
                    t_display = MyDash('Hello')
                    t_display.start()
                # show_fig_sem.release()
                

# async def show_fig():
#     global fig
#     # print('show_fig in ')
#     # app = dash.Dash()
#     # app.run_server(debug=True, use_reloader=False)
#     i=0
#     while True:
#         await asyncio.sleep(1)
#         i=i+1
#         print(i)
#     #     await show_fig_sem.acquire()
#     #     print('show fig')
#     #     app.layout = html.Div([dcc.Graph(figure=fig)])
        




def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

def main():
    print('thread main is running...')
    get_or_create_eventloop().run_until_complete(websockets.serve(echo, 'localhost', 8765))

    get_or_create_eventloop().run_until_complete(SolveQuery())

    get_or_create_eventloop().run_forever()


# asyncio.get_event_loop().run_until_complete(show_fig())


# print('from main thread')
# while True:
#     print('main running')
    # sem.acquire()

if __name__ == '__main__':
    

    main()

