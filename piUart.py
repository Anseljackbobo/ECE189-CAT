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

import serial

# 开启UART2
# ted = serial.Serial(port="/dev/ttyAMA1", baudrate=9600)

# ted.write("Hello World".encode("gbk"))

def init_UART():
    return serial.Serial(port="/dev/ttyAMA1", baudrate=9600)



def send_to_UART(ted, message):
    ted.write(message.encode("ascii"))


def read_byte_UART(ted):
    return ted.read(1)

def read_long_UART(ted, length):
    return ted.read(length)





