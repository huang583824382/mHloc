import asyncio
import websockets
import struct
import cv2
import numpy as np

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
    conf = {'image800x600': 0, 'text': 1, 'poseInfo': 2, 'errorInfo': 3}
    
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
        print('Get Package, type is ', self.conf.keys()[type])
        length = head[2]
        return type, frame[8:]


packageWorker = Package()


async def hello():
    uri = "ws://127.0.0.1:8765"
    async with websockets.connect(uri) as websocket:
        message = packageWorker.generateStream(packageWorker.conf['text'], str.encode('hello'))
        await websocket.send(message)
        img = cv2.imread('datasets/My_lib_dataset/query/room10.jpg')
        data = cv2.imencode('.jpg', img)[1].tobytes()
        print(len(data))
        message = packageWorker.generateStream(packageWorker.conf['image800x600'], data)
        await websocket.send(message)



asyncio.get_event_loop().run_until_complete(hello())
