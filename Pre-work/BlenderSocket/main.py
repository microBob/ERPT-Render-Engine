import socket
import subprocess
import time
import os

HOST = 'localhost'
PORT = 8084

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

# Detect what script to use
scriptFile = ""
if "Pre-work" in os.path.dirname(os.path.realpath(__file__)):
    scriptFile = "/media/microbobu/WorkDrive/Tech/ERPT-Render-Engine/Pre-work/CudaSocket/cmake-build-debug/CudaSocket"
else:
    scriptFile = "./bin/CudaSocket"

print("Start:")
rounds = []
for i in range(4):
    p = subprocess.Popen([scriptFile], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    connection, address = s.accept()

    start = time.time()

    dataBuffer = []
    while True:
        inData = connection.recv(1024)
        if inData:
            dataBuffer.append(inData.decode("utf8"))
        else:
            break

    dataToString = ''.join(dataBuffer).strip()

    dataSplit = dataToString.split(' ')
    pixData = [float(i) for i in dataSplit]

    stop = time.time()

    print("Data Receive Duration:", (stop - start))
    rounds.append((stop - start))

    connection.close()

print("\nAverage:", float(sum(rounds)) / len(rounds))

s.close()
# sendData = "Received: "+dataToString
# connection.sendall(sendData.encode("ascii"))
