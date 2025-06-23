import socket
import struct
import threading
import time


def receive_data(inputData: list, s: socket.socket):
    T = 1 / 50

    while True:
        st = time.time()

        chunk = s.recv(64)
        if chunk == b'':
            raise RuntimeError('socket connection broken')

        # receive linear and angular speed
        x, y, th, wr, wl = struct.unpack('>fffff', chunk)
        inputData[0] = x
        inputData[1] = y
        inputData[2] = th

        et = time.time()
        dt = T - (et - st)
        if dt > 0:
            time.sleep(dt)


def send_data(outputData: list, s: socket.socket):
    T = 1 / 50

    while True:
        st = time.time()

        sent = s.send(struct.pack('>ff', outputData[0], outputData[1]))
        if sent == 0:
            raise RuntimeError('socket connection broken')

        et = time.time()
        dt = T - (et - st)
        if dt > 0:
            time.sleep(dt)


inputSignals = [0, 0, 0, 0]
outputSignals = [-5, 5, 0, 0]


# Connect robot to the socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# wait for connection
print('Wait for connecting')
while True:
    s.connect(('ev3dev.local', 5553))
    print('Connected')
    pubThr = threading.Thread(target=send_data, args=(outputSignals, s))
    subThr = threading.Thread(target=receive_data, args=(inputSignals, s))
    pubThr.start()
    subThr.start()
    break

while True:
    # print(time.time(), outputSignals, inputSignals)
    time.sleep(0.1)
