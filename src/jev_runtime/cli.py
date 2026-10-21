"""Command line entry point."""
import argparse, json, signal, threading
from .service import Runtime, Server

def main(argv=None):
    parser=argparse.ArgumentParser(prog='jev-runtime'); sub=parser.add_subparsers(dest='command',required=True); serve=sub.add_parser('serve'); serve.add_argument('--config',required=True); args=parser.parse_args(argv)
    if args.command=='serve':
        with open(args.config,encoding='utf-8') as stream: config=json.load(stream)
        runtime=Runtime(config); server=Server((runtime.host,runtime.port),runtime); stop=threading.Event()
        def request_stop(*_):
            if not stop.is_set(): stop.set(); threading.Thread(target=server.shutdown,name='jev-shutdown',daemon=True).start()
        signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
        try: server.serve_forever()
        finally: server.server_close()
if __name__=='__main__': main()
