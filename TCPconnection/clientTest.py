import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()  # 获取本地主机名
ip = socket.gethostbyname(host)
port = 9999  # 设置端口号
s.connect((ip, port))
while True:
    message = s.recv(1024)
    print(message.decode('utf-8'))
s.close()