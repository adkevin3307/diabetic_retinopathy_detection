import argparse
import matplotlib.pyplot as plt


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


def show_history(history: dict[str, list]) -> None:
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['accuracy'], label='train_accuracy')

    if ('val_loss' in history) and ('val_accuracy' in history):
        plt.plot(history['val_loss'], label='valid_loss')
        plt.plot(history['val_accuracy'], label='valid_accuracy')

    plt.legend()
    plt.show()
