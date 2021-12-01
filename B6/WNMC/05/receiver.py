import serial
import time

s = serial.Serial('/dev/tty.usbmodem1101', 115200, timeout=1)
time.sleep(2.1)
s.write("r\n".encode())
time.sleep(0.1)
# set logging level
s.write("c[2,0,7]\n".encode())
time.sleep(0.1)
s.write("a[EB]\n".encode())
time.sleep(0.1)
s.write("c[1,0,5]\n".encode())
# set number of retransmissions to 5
time.sleep(0.1)
# wait for settings to be applied
s.write("c[0,1,30]\n".encode())
# set FEC threshold to 30 (apply FEC to packets with payload >= 30)
time.sleep(0.1)
# disable light
s.write("c[0,3,0]\n".encode())
time.sleep(0.1)
#s.write("m[salut\0, DC]\n".encode())
# time.sleep(0.1)
def printlogs():
    print('\n'.join(map(str, s.readlines())))
while True:
    l = input('>>')
    s.write((l+"\n").encode())
    time.sleep(0.1)
    printlogs()