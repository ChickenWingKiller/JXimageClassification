import socket
import threading

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()  # 获取本地主机名
ip = socket.gethostbyname(host)
port = 9999  # 设置端口号
# ip = '127.0.0.1'
serverSocket.bind((ip, port))
serverSocket.listen(5)
clientsocket, addr = serverSocket.accept()
# print(clientsocket)
# print(addr)
# while True:
# #     # clientsocket, addr = serverSocket.accept()
#     print('server started...')
#     message = input()
#     if message == '886':
#         clientsocket.close()
#         break
#     clientsocket.send(message.encode('utf-8'))
# clientsocket.close()

# while(True):
#     clientsocket, addr = serverSocket.accept() #建立客户端连接
#     message = "this is message from server" + "\r\n"
#     clientsocket.send(message.encode("utf-8"))
#     clientsocket.close()
def send():
    while True:
        m = input('请输入：')
        if (m!='886'):
            clientsocket.send(m.encode('utf-8'))
        else:
            clientsocket.send('服务器已断开连接'.encode('utf-8'))
            clientsocket.close()
            break

def receive():
    while True:
        m = clientsocket.recv(1024)
        print(m.decode('utf-8'))

if __name__ == '__main__':
    t_send = threading.Thread(target=send)
    t_receive = threading.Thread(target=receive)
    t_send.start()
    t_receive.start()