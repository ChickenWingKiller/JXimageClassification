import socket
import sys
import threading

# serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #创建socket对象
host = socket.gethostname()  # 获取本地主机名
ip = socket.gethostbyname(host)
# ip = '192.168.130.15'
# # ip = '192.168.88.1'
# # print(host) 汉堡怪兽-R9000P
port = 9999


# # serverSocket.bind((host,port)) #绑定端口号
# serverSocket.bind((ip,port))
# serverSocket.listen(5) #设置最大连接数，超过后排队
#
# while(True):
#     clientsocket, addr = serverSocket.accept() #建立客户端连接
#     # print("链接地址： %s"%str(addr))
#     message = "this is message from server" + "\r\n"
#     clientsocket.send(message.encode("utf-8"))
#     clientsocket.close()

def tcplink(socket, addr):
    print('[%s, %s] is online...' % addr)
    while True:
        try:
            data = socket.recv(1024)
        except:
            socket_set.remove(socket)
            print('[%s, %s] is down!' % addr)
            break
        if data == '886' or not data:
            socket_set.remove(socket)
            socket.close()
            print('[%s, %s] is down!' % addr)
            break
        else:
            list1 = []
            for i in socket_set:
                if i != socket:
                    list1.append(i)
            for i in list1:
                i.send(data)


if __name__ == '__main__':
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_set = set()
    serverSocket.bind((ip, port))
    serverSocket.listen(5)
    print('server is waiting connect......')
    while True:
        clientSocket, addr = serverSocket.accept()
        socket_set.add(clientSocket)
        thread = threading.Thread(target=tcplink, args=(clientSocket, addr))
        thread.start()
        # print(1)
