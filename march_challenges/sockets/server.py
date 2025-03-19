import socket

def run_server():
    # AF.INET states to use Ipv4, and SOCK_STREAM states to use TCP
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Initialises socket variables
    server_ip = '192.168.1.184'
    port = 8000
    # Binds the IP and Port to the socket
    server.bind((server_ip, port))
    # Intialisation of the server to listen for incoming connections
    server.listen(0)
    print(f"Listening on {server_ip}:{port}")
    # Accepts the incoming connection
    client_socket, client_address = server.accept()
    print(f"Accepted connection from {client_address[0]}:{client_address[1]}")

    # Receives the data from the client
    while True:
        request = client_socket.recv(1024)
        request = request.decode("utf-8") # convert bytes to string
        # Close the loop if the client sends a close signal
        if request.lower() == "close":
            client_socket.send("closed".encode("utf-8")) 
            break
        print(f"Received: {request}")
        response = "accepted".encode("utf-8")
        client_socket.send(response)
    # As we are now broken out of the loop, we can close the connection with the client
    client_socket.close()
    print("Connection to client closed")
    server.close()
    
    
run_server()
    