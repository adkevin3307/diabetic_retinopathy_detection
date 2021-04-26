import argparse


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-c', '--csv', type=str, required=True)
    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=10)

    args = parser.parse_args()

    print(args)

    return args
