import socket
import sys

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #创建socket对象
host = socket.gethostname() #获取本地主机名
ip = '192.168.130.15'
# print(host) 汉堡怪兽-R9000P
port = 9999
# serverSocket.bind((host,port)) #绑定端口号
serverSocket.bind((ip,port))
serverSocket.listen(5) #设置最大连接数，超过后排队

while(True):
    clientsocket, addr = serverSocket.accept() #建立客户端连接
    print("链接地址： %s"%str(addr))
    message = "this is message from server" + "\r\n"
    clientsocket.send(message.encode("utf-8"))
    clientsocket.close()