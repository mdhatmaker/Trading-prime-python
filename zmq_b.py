import sys
import zmq

context = zmq.Context()
sock = context.socket(zmq.REP)
#sock.bind(sys.argv[1])
print "binding to socket..."
sock.bind('tcp://127.0.0.1:8080')

while True:
    message = sock.recv()
    sock.send('Echoing: ' + message)
