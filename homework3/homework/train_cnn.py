from .models import CNNClassifier, save_model, load_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
from torchvision import transforms
import torch.utils.tensorboard as tb

from datetime import datetime

def train(args):
    """
    Your code here, modify your HW1 / HW2 code
    """
    from os import path
    def accuracy_cal(data) -> float:
        result_tensor = torch.Tensor(()).to(device)
        for input_b, target_b in data:
            input_b, target_b = input_b.to(device), target_b.to(device)
            out_b = torch.argmax(model(input_b), dim=1)
            comp = target_b == out_b
            result_tensor = torch.cat((result_tensor, comp))

        rs = float(result_tensor.float().mean())
        return rs

    def namer(acc):
        st = str(acc) + '-' + datetime.now().strftime('%d-%m-%Y,%H-%M-%S')
        return st

    print('start training')
    # The device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # The model
    if not args.load:
        model = CNNClassifier().to(device)
    else:
        model = load_model('cnn').to(device)

    # The Loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Hyperparams
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    momentum = args.momentum
    weight_decay = args.weight_decay
    step_size = args.scheduler_step_size
    gamma = args.scheduler_gamma
    verb_step = args.verbose_step

    objective = args.objective
    objective_epoch = None

    bs = args.batch_size

    # Get Data
    train_data = load_data('data/train', batch_size=bs, num_workers=4,
                           transform=transforms.Compose(
                               (
                                   transforms.ToTensor(),
                                   transforms.ColorJitter(0.8, 0.8, 0.8, 0.4),
                                   transforms.RandomHorizontalFlip(p=0.6),
                               )
                           ))
    valid_data = load_data('data/valid', batch_size=bs, num_workers=4)

    # The loss fn
    loss_fn = torch.nn.CrossEntropyLoss()

    # The optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,\
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Accuracy series
    global_step = 0
    epoch = 0
    while epoch < n_epochs:
        # Shuffle batches from the data loader
        for input_b, target_b in train_data:
            input_b, target_b = input_b.to(device), target_b.to(device)
            # Compute the out and the loss
            out = model(input_b)
            loss = loss_fn(out, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log_dir is not None and global_step % 100 == 0:
                train_logger.add_scalar('loss', loss, global_step)

            global_step += 1

        scheduler.step()

        # Evaluate the model
        train_acc = accuracy_cal(train_data)
        valid_acc = accuracy_cal(valid_data)

        if args.log_dir is not None:
            train_logger.add_scalar('accuracy', train_acc, global_step)
            valid_logger.add_scalar('accuracy', valid_acc, global_step)

        if epoch % verb_step == 0:
            print('Epoch %d, Loss %0.4f' % (epoch, float(loss)))
            print('Train accuracy: %0.4f' % train_acc)
            print('Valid accuracy: %0.4f' % valid_acc)

        if valid_acc > objective:
            objective = valid_acc
            objective_epoch = epoch
            save_model(model, namer(valid_acc))

        if epoch == n_epochs - 1:
            if objective_epoch is not None and n_epochs - objective_epoch < args.extension:
                n_epochs += args.extension

        epoch += 1
    # save_model(model)
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-ep', '--n_epochs', type=int)
    parser.add_argument('-mtm', '--momentum', type=float, default=0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-load', type=bool, default=False)
    parser.add_argument('-sch_ss', '--scheduler_step_size', type=int, default=20)
    parser.add_argument('-sch_gamma', '--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('-verb_step', '--verbose_step', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)

    parser.add_argument('-obj', '--objective', type=float, default=0.865)
    parser.add_argument('-ext', '--extension', type=int, default=10)
    parser.add_argument('-trains', '--n_trains', type=int, default=1)

    args = parser.parse_args()
    train(args)
