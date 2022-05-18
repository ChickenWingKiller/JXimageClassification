import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #创建socket对象
host = socket.gethostname() #获取本地主机名
ip = '192.168.130.15'
port = 9999 #设置端口号
# s.connect((host,port)) #连接服务，指定主机和端口
s.connect((ip,port)) #连接服务，指定主机和端口
message = s.recv(1024) #接收小于1024字节的数据
s.close()
print(message.decode('utf-8'))
