import sys
import zmq

context = zmq.context()
sock = context.socket(zmq.REP)
#sock.bind(sys.argv[1])
sock.bind('tcp://localhost:8080')

while True:
    message = sock.recv()
    sock.send('Echoing: ' + message)
