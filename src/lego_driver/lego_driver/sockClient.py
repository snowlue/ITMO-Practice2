import socket
import struct
import threading
import time
from math import cos, pi, sin

import ev3dev2.motor as motor


def receive_data(inputData: list, s: socket.socket):
    T = 1 / 50

    while True:
        st = time.time()

        chunk = s.recv(8)
        if chunk == b'':
            raise RuntimeError('socket connection broken')

        # receive linear and angular speed
        v, w = struct.unpack('>ff', chunk)
        inputData[0] = v
        inputData[1] = w

        et = time.time()
        dt = T - (et - st)
        if dt > 0:
            time.sleep(dt)


def send_data(outputData: list, s: socket.socket):
    T = 1 / 50

    while True:
        st = time.time()

        sent = s.send(struct.pack('>fff', outputData[0], outputData[1], outputData[2]))
        if sent == 0:
            raise RuntimeError('socket connection broken')

        et = time.time()
        dt = T - (et - st)
        if dt > 0:
            time.sleep(dt)


def saturation(vol):
    if vol > 50:
        vol = 50
    elif vol < -50:
        vol = -50
    return vol


def check_angel(angel):
    if angel > pi:
        return angel - 2 * pi
    elif angel + pi < 0:
        return angel + 2 * pi
    else:
        return angel


DOTS = [(1, -1)]
RADIUS = 0.028
BASE = 0.185  # ?
ERROR = 0.05
X, Y = 0, 0
k_s = 3
k_r = 100
inputSignals = [0, 0]
outputSignals = [0, 0, 0]

motorleft = motor.LargeMotor('outB')  # left wheel
motorright = motor.LargeMotor('outA')  # right wheel

startPosleft = motorleft.position * pi / 180
startPosright = motorright.position * pi / 180

lastposleft = motorleft.position * pi / 180
lastposright = motorright.position * pi / 180

x_current, y_current, theta = 0, 0, 0
timeStart = time.time()

v_id = 0
omega_id = 0


try:
    # Connect robot to the socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 5553))
    s.listen(5)

    # wait for connection
    print('Wait for connecting')
    while True:
        clientSock, _ = s.accept()
        pubThr = threading.Thread(target=send_data, args=(outputSignals, clientSock))
        subThr = threading.Thread(target=receive_data, args=(inputSignals, clientSock))
        pubThr.start()
        subThr.start()
        break

    while True:
        posleft = motorleft.position * pi / 180
        posright = motorright.position * pi / 180
        deltaposleft = posleft - lastposleft
        deltaposright = posright - lastposright
        v = (deltaposleft + deltaposright) * RADIUS / 2
        w = (deltaposright - deltaposleft) * RADIUS / BASE

        x_current = x_current + v * cos(theta)
        y_current = y_current + v * sin(theta)
        theta = theta + w

        outputSignals[0] = x_current
        outputSignals[1] = y_current
        outputSignals[2] = theta

        v_id = inputSignals[0]
        omega_id = inputSignals[1]

        # ex = x_goal - x_current
        # ey = y_goal - y_current
        # distance = sqrt(ex*ex + ey*ey)
        # psi = atan2(ey, ex)
        # alpha = check_angel(psi - theta)

        # v_id = k_s * distance
        # omega_id = k_r * alpha

        wr_d = saturation(0.5 * (2 * v_id + BASE * omega_id) / RADIUS)
        wl_d = saturation(0.5 * (2 * v_id - BASE * omega_id) / RADIUS)

        motorright.on(wr_d)
        motorleft.on(wl_d)

        lastposleft = posleft
        lastposright = posright

except Exception:
    print('Error')
finally:
    motorleft.stop()
    motorright.stop()
