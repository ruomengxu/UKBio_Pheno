import socket
import getpass

def get_output_dir():
    username = getpass.getuser()

    if is_server():
        return '/home/{}'.format(username)
    else:
        return '../data'

def is_server():
    hostname = socket.gethostname()

    if hostname == 'ip-10-0-0-87':
        return True
    else:
        return False
