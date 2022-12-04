import argparse

from model_api.controller import app

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, help='port')
parser.add_argument('--host', type=str, help='host')

if __name__ == "__main__":
    args = parser.parse_args()
    app.run(port=args.port, host=args.host)
