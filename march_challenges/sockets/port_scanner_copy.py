import socket
from colorama import Fore, init
import argparse

port_list = [443]
host = '192.168.1.72'
port_counter = 0

def check_port(port, host) -> bool: 
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.settimeout(0.1)  # 5-second timeout
    try:
        server.connect((host, port))
        print(f"Port {port} | {Fore.GREEN} OPEN {Fore.RESET}")
        banner = server.recv(1024).decode().strip()
        print(f"Service Running: {banner} ")
        server.close()
        return True
    except ConnectionRefusedError:
        print(f"Port {port} | {Fore.RED} CLOSED {Fore.RESET}")
        return False
    except socket.timeout:
        print(f"Port {port} | {Fore.RED} CLOSED {Fore.RESET}")
        return False
    
for port in port_list:
    if check_port(port, host):
        port_counter += 1

print(f"All Ports checked | {port_counter} Port's Open")