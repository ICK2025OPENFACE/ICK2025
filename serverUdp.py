import socket

host = "127.0.0.1"
port = 4242

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_socket.bind((host, port))
print(f"Serwer UDP nasłuchuje na {host}:{port}")

try:
    while True:
        data, addr = server_socket.recvfrom(1024)
        print(f"Otrzymano wiadomość od {addr}: {data.decode('utf-8')}")

except KeyboardInterrupt:
    print("Serwer zatrzymany.")
finally:
    server_socket.close()
