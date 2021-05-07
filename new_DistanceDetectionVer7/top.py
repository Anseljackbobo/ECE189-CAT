import cv2
import pyftdi.serialext
import numpy as np
import multiprocessing
from newDistanceVer7_prog import *
from multiprocessing import Process, Manager, Value

# init uart
port = pyftdi.serialext.serial_for_url('ftdi://ftdi:232:AB0KT24J/1', baudrate=3000000)
port.timeout = 0.06


manager = Manager()

return_val = Value('i', 0)
worker_is_free = manager.Value('i', 0)

# Window name in which image is displayed
window_name = 'Image'
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
org2 = (50, 100)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

return_val.value = 0
worker_is_free.value = 1

message_slave = "slave message not yet arrived"
while True:
    if worker_is_free.value==1:
        worker_is_free.value = 0
        process_master=multiprocessing.Process(target=grand_worker,args=(return_val,worker_is_free))
        process_master.start()

    if(return_val.value == 1):
        message = "Blocked"
    else:
        message = "Released"

    # print(message)

    image = np.zeros((480, 640))
    image = cv2.putText(image, message, org, font, fontScale, color, thickness, cv2.LINE_AA)



    # receive from the other:
    data = port.read(1)
    if(data==b'b'):
        message_slave = "Slave Blocked"
        port.write('f')
    elif(data==b'r'):
        message_slave = "Slave Release"
        port.write('f')
    else:
        pass
        # message_slave = "not arrived"

    image = cv2.putText(image, message_slave, org2, font, fontScale, color, thickness, cv2.LINE_AA)











    # Displaying the image
    cv2.imshow(window_name, image) 
    cv2.waitKey(1)