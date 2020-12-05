import socket

HOST = 'localhost'
PORT = 8083

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

connection, address = s.accept()

data = connection.recv(1024)
dataToString = repr(data)
print("Received:", dataToString)
sendData = "Received: "+dataToString
connection.sendall(sendData.encode("ascii"))
