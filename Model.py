import numpy as np
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience, moniter):
        self._patience = patience
        self.moniter = moniter
        self._counter = 0
        self._best_score = None
        self.early_stop = False

    def __call__(self, score) -> bool:
        if self._best_score == None:
            self._best_score = score

        if 'loss' in self.moniter:
            if score > self._best_score:
                self._counter += 1

                if self._counter >= self._patience:
                    self.early_stop = True
            else:
                self._best_score = score
                self._counter = 0
        elif 'accuracy' in self.moniter:
            if score < self._best_score:
                self._counter += 1

                if self._counter >= self._patience:
                    self.early_stop = True
            else:
                self._best_score = score
                self._counter = 0

        return self.early_stop


class Model:
    def __init__(self, net, optimizer, criterion):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = self.net.to(self.device)

    def train(self, train_loader, epochs, val_loader=None, scheduler=None, early_stopping=None):
        history = defaultdict(list)

        length = len(str(epochs))

        for epoch in range(epochs):
            self.net.train()

            loss = 0.0
            accuracy = 0
            for i, (x_train, y_train) in enumerate(train_loader):
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()

                output = self.net(x_train)
                _, y_hat = output.max(dim=1)
                accuracy += (y_hat == y_train).sum().item()

                temp_loss = self.criterion(output, y_train)
                loss += temp_loss.item()

                temp_loss.backward()
                self.optimizer.step()

                current_progress = (i + 1) / len(train_loader) * 100
                progress_bar = '=' * int((i + 1) * (20 / len(train_loader)))
                print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%, loss: {temp_loss.item():.3f}', end='')

            loss /= len(train_loader)
            accuracy /= len(train_loader.dataset)

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

            print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{"=" * 20}], loss: {loss:.3f}, accuracy: {accuracy:.3f}', end='' if val_loader else '\n')

            if val_loader:
                val_history = self.test(val_loader, verbose=False)

                history['val_loss'].append(val_history['loss'])
                history['val_accuracy'].append(val_history['accuracy'])

                if early_stopping:
                    if early_stopping(history[early_stopping.moniter]):
                        print('Sweet Point')
                        break

                print(f', val_loss: {history["val_loss"][-1]:.3f}, val_accuracy: {history["val_accuracy"][-1]:.3f}')

            if scheduler:
                scheduler.step()

        return dict(history)

    def test(self, test_loader, verbose=True):
        self.net.eval()

        history = {}

        loss = 0.0
        accuracy = 0
        predict = []
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(test_loader):
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                output = self.net(x_test)
                _, y_hat = output.max(dim=1)
                accuracy += (y_hat == y_test).sum().item()

                predict.append(y_hat)

                temp_loss = self.criterion(output, y_test).item()
                loss += temp_loss

                if verbose:
                    current_progress = (i + 1) / len(test_loader) * 100
                    progress_bar = '=' * int((i + 1) * (20 / len(test_loader)))
                    print(f'\rTest: [{progress_bar:<20}] {current_progress:6.2f}%, loss: {temp_loss:.3f}', end='')

        loss /= len(test_loader)
        accuracy /= len(test_loader.dataset)

        history['loss'] = loss
        history['accuracy'] = accuracy
        history['predict'] = predict

        if verbose:
            print(f'\rTest: [{"=" * 20}], test_loss: {loss:>.3f}, test_accuracy: {accuracy:.3f}')

        return history

    def summary(self, input_shape):
        hooks = []
        summary_dict = OrderedDict()

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_index = len(summary_dict)

                key = f'{class_name}-{module_index + 1}'
                summary_dict[key] = {}
                summary_dict[key]['input_shape'] = list(input[0].shape)
                summary_dict[key]['output_shape'] = list(output[0].shape)

                train_parameters = 0
                total_parameters = 0
                for parameter in module.parameters():
                    amount = np.prod(parameter.shape)

                    if parameter.requires_grad:
                        train_parameters += amount
                    total_parameters += amount
                summary_dict[key]['train_parameters'] = train_parameters
                summary_dict[key]['total_parameters'] = total_parameters

            if (not isinstance(module, nn.Sequential)) and (not isinstance(module, nn.ModuleList)) and (not (module == self.net)):
                hooks.append(module.register_forward_hook(hook))

        self.net.apply(register_hook)

        input_data = torch.randn(input_shape).to(self.device)
        self.net(input_data)

        for i in hooks:
            i.remove()

        print('-' * 90)
        print(f'{"Layer":>25}{"Input Shape":>25}{"Output Shape":>25}{"Parameters":>15}')
        print('=' * 90)

        train_parameters = 0
        total_parameters = 0
        for key, value in summary_dict.items():
            print(f'{key:>25}{str(value["input_shape"]):>25}{str(value["output_shape"]):>25}{value["train_parameters"]:>15}')

            train_parameters += value['train_parameters']
            total_parameters += value['total_parameters']

        print('=' * 90)
        print(f'Train Parameters: {train_parameters}, Total Parameters: {total_parameters}')
        print('-' * 90)

    def save(self, save_path: str):
        torch.save(self.net, save_path)
