import serial
import time
import unireedsolomon as rs


s = serial.Serial('/dev/tty.usbmodem1201', 115200, timeout=1)
time.sleep(2)
counter=0
res=''
tmp=""
final= ""
acc=""

while acc[-8:]!= '11111111':
    tmp = s.read(1).decode()
    acc += tmp
    print(acc)
    s.flush()
    
while True:
    #s.read_until(expected=str.encode('165'))
    tmp = s.read(1).decode()
    res =res+tmp
    print(tmp.encode())
    
    if tmp =="0":
        counter+=1
        print("counter:"+str(counter))
        
    elif tmp =="\n" or tmp==" " or tmp=="" or "\r":
        print("newline")
        counter=0
    else :
        print("tmp:"+tmp)
        counter=0

    if counter>=8:
        break
for i in range(0,len(res),8):
   final+=res[i:i+8]

print(final)
list =[]
for i in range(0,len(final),8):
    list.append(final[i:i+8])

for l in list:
    print((chr(int(l, 2))))

# write to the device’s serial port 
# while 1:      #Do this in loop
    
#     print(s.read(1000))
    # var = input() #get input from user
   
    
    # if (var == '1'): #if the value is 1
    #     s.write(str.encode('1')) #send 1
    #     s.write(str.encode('a[AB]\n')) #send new line
    #     print ("LED turned ON")
    #     time.sleep(1)
    
    # if (var == '0'): #if the value is 0
    #     s.write(str.encode('0')) #send 0
    #     w = s.read(10)#send new line
    #     print ("LED turned OFF"+ " "+ str(w, 'utf-8'))
    #     time.sleep(1)


# while 1:
#     s.write(b'm[hello world!\0,CD]\n') #send message to device with address CD
#     time.sleep(0.5)