import socket

def run_client():
    # Initialise socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Establish server socket details
    server_ip = '192.168.1.72'
    server_port = 8000
    # Connect to the server
    client.connect((server_ip, server_port))
    # Start loop to send message to the server
    while True:
        # input message and send it to the server
        msg = input("Enter message: ")
        client.send(msg.encode("utf-8")[:1024])
        
        response = client.recv(1024)
        response = response.decode("utf-8")
        
        if response.lower() == "closed":
            break
        
        print(f"Received: {response}")
    
    client.close()
    print("Server has closed the connection")
    
run_client()