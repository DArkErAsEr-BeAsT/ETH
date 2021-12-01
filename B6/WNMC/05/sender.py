#!/usr/bin/python
# coding: utf-8
import serial
import time
import sys


s = serial.Serial("/dev/tty.usbmodem1101", timeout=1)
time.sleep(2)
s.baudrate = 115200
while True:
    s.write("r\n".encode())
    time.sleep(0.1)
    # write to the device’s serial port 
    s.write("a[DC]\n".encode()) #set the device address to AB
    time.sleep(0.1)
    #s.write('a\n'.encode())
    time.sleep(0.1)
    s.write("c[2,0,7]\n".encode())
    time.sleep(0.1)

    s.write("c[1,0,5]\n".encode()) #set number of retransmissions to 5
    time.sleep(0.1) #wait for settings to be applied
    s.write("c[0,1,30]\n".encode()) #set FEC threshold to 30 (apply FEC to packets with payload >= 30)
    time.sleep(0.1) #wait for settings to be applied

    time.sleep(0.1) #wait for settings to be applied
    s.write("m[mjsn\0,EB]\n".encode()) 
    time.sleep(0.1)
    with open("tmp1.txt", "a") as f:
        f.write(str(s.readlines()))

    f1 = open("tmp1.txt", "r+")
    input = f1.read()
    input=input.replace(',','\n')
    f2=open("tmp1.txt","w+")
    f2.write(input)
    print("sent")
#send message to device with address CD 
# s.write("r\n".encode())
#s.read(1)
# time.sleep(0.1) #wait for settings to be applied
# s.write(b"c[1,0,5]\n") #set number of retransmissions to 5
# time.sleep(0.1) #wait for settings to be applied
# s.write(b"c[0,1,30]\n") #set FEC threshold to 30 (apply FEC to packets with payload >= 30)
# time.sleep(0.1) #wait for settings to be applied
s#.write(str.encode("m[hello world!\0,CD]\n")) #send message to device with address CD 
# message = "" 
# while True: #while not terminated 
#  try: 
#    byte = s.read(1).decode() #read one byte (blocks until data available or timeout reached) 
#    if byte=='\n': #if termination character reached
#      print(message )#print message
#      message = "" #reset message
#    else:
#      message = message + byte #concatenate the message 
#  except serial.SerialException: 
#    continue #on timeout try to read again 
#  except KeyboardInterrupt: 
#    sys.exit() #on ctrl-c terminate program