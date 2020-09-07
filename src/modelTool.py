import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from .utils.progressBar import ProgressBar
from .dataTool import iDataset


class BaseModel(abc.ABC):
    '''Abstract base class for Network models.

    Attributes:
        model (torch.nn.module) : A Pytorch Model.
        model_name (str) : The name of Model.
    '''
    def __init__(
        self,
        model,
        model_name,
    ):
        self._model = model
        self._model_name = model_name

    def name(self):
        return self.BaseModel.__name__


class ModelTool(BaseModel):
    '''Model Tool。
    Some tools for auto train model.

    Attributes:
        model (torch.nn.module) : A pytorch model.
        model_name (str) : The name of model.
        file_path (str) : The path where model saved.
        criterion (torch.nn, optional) : If 'None', this tools will auto make a nn.CrossEntropyLoss() to train model.
        optimizer (torch.optim, optional) : if 'None', this tools will auto build a optim.SGD(self._model.parameters(),
            lr=0.05, weight_decay=5e-4) to train model.
        best_accuracy (float, optional) : This attribute is used to resume model. If a accuracy in any step over
            best_accuracy, tools will auto save model to 'file_path' and record accuracy.
        epoch (int, optional) : Same like above.

    Example:
        >>> model = torchvision.model.vgg16();
        >>> mt = ModelTool(model, 'vgg16', './checkpoint/vgg16.pth')
        >>> mt.auto_train(train_loader, test_loader)
    '''
    def __init__(
        self,
        model,
        model_name,
        file_path,
        criterion=None,
        optimizer=None,
        best_accuracy=0.,
        epoch=0,
    ):
        super().__init__(model, model_name)
        self._file_path = file_path
        self._criterion = criterion
        self._optimizer = optimizer
        self._best_accuracy = best_accuracy
        self._epoch = epoch
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def resume(self, file_path=None):
        """Resume model state from file_path.
        """
        if file_path is None:
            file_path = self._file_path
        print('===> Resume model from :', file_path)
        state = torch.load(file_path, map_location=self._device)
        self._model.load_state_dict(state['model'])
        self._best_accuracy = state['acc']
        self._epoch = state['epoch']
        self._criterion = state['criterion']
        self._optimizer = state['optimizer']

    def save(self, file_path=None):
        """Save Model to file_path.
        """
        if file_path is None:
            file_path = self._file_path
        print('===> Save model to ', file_path)
        if isinstance(self._model, torch.nn.DataParallel):
            st = self._model.module.state_dict()
        else:
            st = self._model.state_dict()
        state = {
            'model': st,
            'model_name': self._model_name,
            'acc': self._best_accuracy,
            'epoch': self._epoch,
            'criterion': self._criterion,
            'optimizer': self._optimizer,
        }
        torch.save(state, file_path)

    def auto_train(self,
                   train_loader,
                   test_loader,
                   epoch_max=200,
                   lr_alpha=0.3,
                   lr_beta=0.7,
                   verbose=True):
        """Auto train.

        Attributes:
            train_loader (torch.utils.data.DataLoader) : The DataLoader of train set.
            test_loader (torch.utils.data.DataLoader) : The DataLoader of test set.
            epoch_max (int, optional) : The max epoch of training model.
            lr_alpha (float, optional) : The alpha of learning rate.
            lr_beta (float, optional) : The beat of learning rate.
            verbose (bool, optional) : If true, the message of training process will show in terminal.

        """
        self._model = self._model.to(self._device)
        if self._device == 'cuda':
            self._model = nn.DataParallel(self._model)

        if self._criterion is None:
            self._criterion = nn.CrossEntropyLoss()

        if self._optimizer is None:
            self._optimizer = optim.SGD(self._model.parameters(),
                                        lr=0.05,
                                        weight_decay=5e-4)

        scheduler = lr_scheduler.ReduceLROnPlateau(self._optimizer)

        for self._epoch in range(self._epoch + 1, self._epoch + epoch_max):
            print('Epoch: ', self._epoch)
            train_loss = self._train_step(train_loader, self._criterion,
                                          self._optimizer, verbose)
            test_loss = self._test_step(test_loader, self._criterion,
                                        self._optimizer, verbose)

            acc = 100 * test_loss['correct'] / test_loss['total']
            acc_t = 100 * train_loss['correct'] / test_loss['total']
            scheduler.step(acc_t * lr_alpha + acc * lr_beta)
            if self._best_accuracy < acc:
                self._best_accuracy = acc
                self.save(self._file_path)

    def _test_step(self,
                   data_loader,
                   criterion=None,
                   optimizer=None,
                   verbose=True):
        model = self._model
        model.eval()
        total_loss, correct, total = 0, 0, 0
        device = self._device

        if criterion is None:
            criterion = self._criterion

        if optimizer is None:
            optimizer = self._optimizer

        pb = ProgressBar()
        pb.start()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if verbose:
                msg = 'Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})'.format(
                    total_loss / (batch_idx + 1), 100. * correct / total,
                    correct, total)

                pb.print(batch_idx + 1, len(data_loader), msg)

        return {'loss': total_loss, 'correct': correct, 'total': total}

    def _train_step(self,
                    data_loader,
                    criterion=None,
                    optimizer=None,
                    verbose=True):
        model = self._model
        model.train()
        total_loss, correct, total = 0, 0, 0
        device = self._device

        pb = ProgressBar()
        pb.start()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if verbose:
                msg = 'Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})'.format(
                    total_loss / (batch_idx + 1), 100. * correct / total,
                    correct, total)

                pb.print(batch_idx + 1, len(data_loader), msg)

        return {'loss': total_loss, 'correct': correct, 'total': total}

    def __str__(self):
        return '*-------------*\nModel Name: ' + self._model_name + '\nAccuracy: ' + str(
            self._best_accuracy) + '\nEpoch: ' + str(
                self._epoch) + '*-------------*'

    def add_forward_hook(self, hook_function=None):
        def save_activation(name):
            ''' Creates hooks to the activations
            Args:
                name (string): Name of the layer to hook into
            '''
            def hook(mod, inp, out):
                ''' Saves the activation hook to dictionary
                '''
                if name not in self.bottlenecks_tensors.keys():
                    self.bottlenecks_tensors[name] = []
                self.bottlenecks_tensors[name].append(out)

            return hook

        handles = []
        for name, mod in self.model.named_modules():
            if (self.bottlenecks is not None) and (name
                                                   not in self.bottlenecks):
                continue
            if hook_function is None:
                handle = mod.register_forward_hook(save_activation(name))
            else:
                handle = mod.register_forward_hook(hook_function(name))
            handles.append(handle)
        return handles

    def run_examples(self, examples, transform, verbose=True):
        ''' Generating the activations and ends from the images.

        Attributes:
            examples (array) : a numpy class which storing images, e.g. [32, 32, 3].
            transform (torchvision.transforms) : A transform from torchvision.transforms.
        '''
        # Initialize the tensor Dict
        model = self._model
        bottlenecks_tensors = {}
        ends = []
        inputs = []
        handles = self.add_forward_hook()

        model.to(self._device)

        concept_dataset = iDataset(images=examples, transform=transform)
        concept_loader = DataLoader(concept_dataset)
        for idx, batch in enumerate(concept_loader):
            batch = Variable(batch, requires_grad=True)
            batch = batch.to(self._device)
            inputs.append(batch)
            # need to run batch through the model to capture activations
            ends.append(model(batch))
            if verbose:
                print("[{}/{}]".format(idx + 1, len(concept_loader)))
            if idx > 50:
                break

        for handle in handles:
            handle.remove()

        return bottlenecks_tensors, ends, inputs

    def eval(self, *args, **kwargs):
        """ Sets wrapped model to eval mode as is done in pytorch.
        """
        self.model.eval(*args, **kwargs)

    def train(self, *args, **kwargs):
        """ Sets wrapped model to train mode as is done in pytorch.
        """
        self.model.train(*args, **kwargs)

    def __call__(self, x):
        """ Calls prediction on wrapped model pytorch.
        """
        ends = self._model(x)
        return ends

    def label_to_id(self, label):
        """Convert label (string) to index in the logit layer (id).
        Override this method if label to id mapping is known. Otherwise,
        default id 0 is used.
        """
        print('label_to_id undefined. Defaults to returning 0.')
        return 0

    def id_to_label(self, idx):
        """Convert index in the logit layer (id) to label (string).
        Override this method if id to label mapping is known.
        """
        print('id_to_label undefined. Defaults to returning 0.')
        return '0'
