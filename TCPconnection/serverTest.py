import socket

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()  # 获取本地主机名
ip = socket.gethostbyname(host)
port = 9999  # 设置端口号
serverSocket.bind((ip, port))
serverSocket.listen(5)
clientsocket, addr = serverSocket.accept()
while True:
#     # clientsocket, addr = serverSocket.accept()
    print('server started...')
    message = input()
    if message == '886':
        clientsocket.close()
        break
    clientsocket.send(message.encode('utf-8'))
clientsocket.close()

# while(True):
#     clientsocket, addr = serverSocket.accept() #建立客户端连接
#     message = "this is message from server" + "\r\n"
#     clientsocket.send(message.encode("utf-8"))
#     clientsocket.close()