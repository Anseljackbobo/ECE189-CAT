# 参考了https://blog.csdn.net/weixin_40796925/article/details/107907991
# 展示所有串口命令 dtoverlay -a | grep uart
# 查看特定串口信息 dtoverlay -h uart2
# 各 UART 串口与 GPIO 对应关系
# GPIO14 = TXD0 -> ttyAMA0
# GPIO0  = TXD2 -> ttyAMA1
# GPIO4  = TXD3 -> ttyAMA2
# GPIO8  = TXD4 -> ttyAMA3
# GPIO12 = TXD5 -> ttyAMA4
# GPIO15 = RXD0 -> ttyAMA0
# GPIO1  = RXD2 -> ttyAMA1
# GPIO5  = RXD3 -> ttyAMA2
# GPIO9  = RXD4 -> ttyAMA3
# GPIO13 = RXD5 -> ttyAMA4

# 用UART2, GPIO 0 as TXD2; GPIO 1 as RXD2

import pyftdi.serialext

# 开启UART2
# ted = serial.Serial(port="/dev/ttyAMA1", baudrate=9600)

# ted.write("Hello World".encode("gbk"))

def init_UART():
    pass
    # port = pyftdi.serialext.serial_for_url('ftdi://ftdi:232:AB0KT6DJ/1', baudrate=9600)
    # ted = serial.Serial(port="/dev/ttyTHS1", baudrate=9600)
    # send_to_UART(ted, "Initiated\n")
    # return port



def send_to_UART(port, message):
    pass
    # message = message.strip()
    # print("sending "+message+"\n")
    # port.write(message.encode("ascii"))
    # port.write('\n'.encode("ascii"))


def read_byte_UART(port):
    pass
    # return port.read(1)

def read_long_UART(port, length):
    return port.read(length)





