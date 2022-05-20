import socket
import sys
import threading

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #创建socket对象
host = socket.gethostname()  # 获取本地主机名
ip = socket.gethostbyname(host)
# ip = '192.168.130.15'
# # ip = '192.168.88.1'
port = 9999  # 设置端口号
# # s.connect((host,port)) #连接服务，指定主机和端口
# s.connect((ip,port)) #连接服务，指定主机和端口
#
# message = s.recv(1024) #接收小于1024字节的数据
# s.close()
# print(message.decode('utf-8'))
# from click._compat import raw_input

name = input('input your name:')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ip, port))


def recv_message():
    while True:
        try:
            data = s.recv(1024)
            print(data.decode('utf-8') + '\r')
        except:
            break


if __name__ == '__main__':
    thread = threading.Thread(target=recv_message)
    thread.start()
    while True:
        message = input()
        # message = message.encode("utf-8")
        if message == '886':
            s.send(message)
            s.close()
            break
        s.send((name + ":" + message).encode("utf-8"))
