import socket
import threading

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()  # 获取本地主机名
ip = socket.gethostbyname(host)
# ip = '192.168.231.10'
# ip = '192.168.43.35'
# ip = '127.0.0.1'
port = 9999  # 设置端口号
s.connect((ip, port))
# s.send('123'.encode('utf-8'))
# print(s.recv(1024).decode('utf-8'))
# while True:
#     message = s.recv(1024)
##     print(type(message)) #bytes
#     print(message)
#     print(message.decode('utf-8'))
# s.close()

def send():
    while True:
        m = input('请输入：')
        if (m != '886'):
            s.send(m.encode('utf-8'))
        else:
            s.send('客户端已断开连接'.encode('utf-8'))
            s.close()
            break
def receive():
    while True:
        m = s.recv(1024)
        print(m.decode('utf-8'))

if __name__ == '__main__':
    t_send = threading.Thread(target=send)
    t_receive = threading.Thread(target=receive)
    t_send.start()
    t_receive.start()