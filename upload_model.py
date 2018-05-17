import argparse

from mlboardclient.api import client


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        required=True
    )
    parser.add_argument(
        '--version',
        default='1.0.0',
        required=True
    )
    parser.add_argument(
        '--from-path',
        required=True
    )

    args = parser.parse_args()

    mlboard = client.Client()
    mlboard.model_upload(args.model_name, args.version, args.from_path)
