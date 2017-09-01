#!/usr/bin/python
#coding=utf-8
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import cgi
import API as qapi

PORT_NUMBER = 8080

api = qapi.API()
q = ['','']

#This class will handles any incoming request from
#the browser
class myHandler(BaseHTTPRequestHandler):
    # Handler for the GET requests

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"

        try:
            # Check the file extension required and
            # set the right mime type

            sendReply = False
            if self.path.endswith(".html"):
                mimetype = 'text/html'
                sendReply = True
            if self.path.endswith(".jpg"):
                mimetype = 'image/jpg'
                sendReply = True
            if self.path.endswith(".gif"):
                mimetype = 'image/gif'
                sendReply = True
            if self.path.endswith(".js"):
                mimetype = 'application/javascript'
                sendReply = True
            if self.path.endswith(".css"):
                mimetype = 'text/css'
                sendReply = True

            if sendReply == True:
                # Open the static file requested and send it
                f = open(curdir + sep + self.path)
                self.send_response(200)
                self.send_header('Content-type', mimetype)
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
            return

        except IOError:
            self.send_error(404, 'File Not Found: %s' % self.path)

    def do_POST(self):

            if self.path == "/segment":
                global q
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)


                if post_data[-1]=='\n':
                    post_data=post_data[:-1]
                post_data = post_data.split('\n')
                if len(post_data)<3:
                    post_data=['','',post_data[-1].strip('\n').strip('\r')]

                self.send_response(200)
                self.end_headers()
                if post_data[-1] == '':
                    self.wfile.write('')
                    return

                post_data = post_data[-3:]

                res = api.query(post_data)

                self.wfile.write(('\n'.join(post_data)+'\n'.encode('utf-8')+''.join(res).encode('utf-8')))



            return


try:
	#Create a web server and define the handler to manage the
	#incoming request
	server = HTTPServer(('', PORT_NUMBER), myHandler)
	print 'Started httpserver on port ' , PORT_NUMBER

	#Wait forever for incoming htto requests
	server.serve_forever()

except KeyboardInterrupt:
	print '^C received, shutting down the web server'
	server.socket.close()
