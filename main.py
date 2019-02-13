import os
import sys
import string
import random
import argparse
import base64
import imghdr

import tornado.web
import tornado.ioloop
import tornado.options
import tornado.httpserver
from tornado.options import define, options

from classify_image import maybe_download_and_extract, run_inference_on_image

uploads_dir = 'uploads'
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)


def create_flags():
    """"
    Copy pasted from original classify_image
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    return parser.parse_known_args()

FLAGS, unparsed = create_flags()
maybe_download_and_extract(FLAGS)


class Application(tornado.web.Application):
    def __init__(self, anaconda_project_hosts):
        handlers = [
            (r"/", IndexHandler),
            (r"/upload", UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers, debug=True)
        self._application = dict(anaconda_project_hosts=anaconda_project_hosts)

class PrepHandler(tornado.web.RequestHandler):
    def prepare(self):
        anaconda_project_hosts = self.application._application['anaconda_project_hosts']
        if self.request.host not in anaconda_project_hosts:
            print("{} not allowed in aborting...".format(self.request.host))
            print("Allowed hosts: {}".format(anaconda_project_hosts))
            raise tornado.web.HTTPError(403)


class IndexHandler(PrepHandler):
    def get(self):
        self.render("upload_form.html")


class UploadHandler(PrepHandler):
    def post(self):
        file1 = self.request.files['file1'][0]
        original_fname = file1['filename']
        file_body = file1['body']
        extension = imghdr.what('', file_body)
        if extension not in ALLOWED_EXTENSIONS:
            print("Not a valid filetype")
            self.redirect("/")
        else:
            fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
            final_filename = fname+'.'+extension
            file_path = os.path.join(uploads_dir, final_filename)
            output_file = open(file_path, 'wb')
            output_file.write(file_body)
            output_file.close()
            img = base64.b64encode(file_body)
            values = run_inference_on_image(file_path, FLAGS)
            self.render('prediction.html', filename=final_filename, img=img, values=values)


def main(anaconda_project_hosts):
    http_server = tornado.httpserver.HTTPServer(Application(anaconda_project_hosts))
    http_server.listen(options.port, address=options.address)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    # arg parser for the standard anaconda-project options
    parser = argparse.ArgumentParser(prog="imagenet-flask",
                            description="Classification Webapp with Tensorflow")
    parser.add_argument('--anaconda-project-host', action='append', default=[],
                        help='Hostname to allow in requests')
    parser.add_argument('--anaconda-project-port', action='store', default=8086, type=int,
                        help='Port to listen on')
    parser.add_argument('--anaconda-project-iframe-hosts',
                        action='append',
                        help='Space-separated hosts which can embed us in an iframe per our Content-Security-Policy')
    parser.add_argument('--anaconda-project-no-browser', action='store_true',
                        default=False,
                        help='Disable opening in a browser')
    parser.add_argument('--anaconda-project-use-xheaders',
                        action='store_true',
                        default=False,
                        help='Trust X-headers from reverse proxy')
    parser.add_argument('--anaconda-project-url-prefix', action='store', default='',
                        help='Prefix in front of urls')
    parser.add_argument('--anaconda-project-address',
                        action='store',
                        default='0.0.0.0',
                        help='IP address the application should listen on.')

    args = parser.parse_args(sys.argv[1:])
    anaconda_project_hosts = args.anaconda_project_host
    anaconda_project_port = args.anaconda_project_port
    anaconda_project_address = args.anaconda_project_address

    if len(anaconda_project_hosts) == 0:
        local_hosts = ['localhost', '127.0.0.1']
        anaconda_project_hosts = ['{}:{}'.format(host, anaconda_project_port) for host in local_hosts]

    define("port", default=anaconda_project_port, help="run on the given port", type=int)
    define("address", default=anaconda_project_address, help="IP address the application should listen on.", type=str)
    print("Tornado App running on {}:{} with accepted host: {}".format(
        anaconda_project_address,
        anaconda_project_port,
        anaconda_project_hosts))
    main(anaconda_project_hosts)
