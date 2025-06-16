import time
from math import pi, cos, sin, atan2, sqrt
import threading
import socket
import struct

def receive_data(inputData: list, s: socket.socket):
    T = 1/50

    while True:
        st = time.time()

        chunk = s.recv(8)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
    
        # receive linear and angular speed
        v, w = struct.unpack(">ff", chunk)
        inputData[0] = v
        inputData[1] = w

        et = time.time()
        dt = T - (et - st)
        if dt > 0:
            time.sleep(dt)

def send_data(outputData: list, s: socket.socket):
    T = 1/50

    while True:
        st = time.time()

        sent = s.send(struct.pack(">fff", outputData[0], outputData[1], outputData[2]))
        if sent == 0:
            raise RuntimeError("socket connection broken")

        et = time.time()
        dt = T - (et - st)
        if dt > 0:
            time.sleep(dt)


inputSignals = [0, 0, 0, 0]
outputSignals = [1, 2, 3, 0]



# Connect robot to the socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 8089))
s.listen(5)

# wait for connection
print("Wait for connecting")
while True:
    clientSock, _ = s.accept()
    print("Connected")
    pubThr = threading.Thread(target=send_data, args=(outputSignals,clientSock))
    subThr = threading.Thread(target=receive_data, args=(inputSignals,clientSock))
    pubThr.start()
    subThr.start()
    break

while True:
    print(time.time(), outputSignals, inputSignals)
    time.sleep(0.1)