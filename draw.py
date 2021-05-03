import glob
import matplotlib.pyplot as plt


def show(net: str) -> dict[str, float]:
    max_accuracy = {}

    plt.figure(figsize=(12, 8))

    history_files = glob.glob(f'scores/score_{net}_*.txt')
    print(history_files)

    for history_file in history_files:
        with open(history_file, 'r') as txt_file:

            train_acc = list(map(lambda x: float(x), txt_file.readline().strip()[1: -1].split(', ')))
            test_acc = list(map(lambda x: float(x), txt_file.readline().strip()[1: -1].split(', ')))

            key = history_file.split('.')[0].split('_')[-1]
            max_accuracy[key] = max(test_acc)

            if key == 'True':
                key = 'Pretrained'
            if key == 'False':
                key = 'From Scratch'

            plt.plot(train_acc, label=f'{key} Train')
            plt.plot(test_acc, label=f'{key} Test')

    plt.legend()
    plt.show()

    return max_accuracy


if __name__ == '__main__':
    max_accuracy = {}

    max_accuracy['resnet18'] = show('resnet18')
    max_accuracy['resnet50'] = show('resnet50')

    print(max_accuracy)

    print()
    print(f'\t{" " * 15} | {"Pretrained":^20} | {"From Scratch":^20} |')
    print(f'\t{"-" * 63}')
    print(f'\t{"ResNet18":<15} | {max_accuracy["resnet18"]["True"]:^20.3f} | {max_accuracy["resnet18"]["False"]:^20.3f} |')
    print(f'\t{"ResNet50":<15} | {max_accuracy["resnet50"]["True"]:^20.3f} | {max_accuracy["resnet50"]["False"]:^20.3f} |')
    print()
