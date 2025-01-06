from .models import CNNClassifier, save_model, load_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb

from datetime import datetime
from os import path

def train(args):
    def accuracy_cal(data) -> float:
        result_tensor = torch.Tensor(())
        for input_b, target_b in data:
            out_b = torch.argmax(model(input_b), dim=1)
            comp = target_b == out_b
            result_tensor = torch.cat((result_tensor, comp))

        rs = float(result_tensor.float().mean())
        return rs

    def namer(acc):
        st = str(acc) + ' - ' + datetime.now().strftime('%d/%m/%Y, %H/%M/%S')
        return st

    print('start training')

    # The model
    if not args.load:
        model = CNNClassifier(layers=(8, 8, 16, 32))
    else:
        model = load_model()

    # The Loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

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

    # Get Data
    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

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
            # Compute the out and the loss
            out = model(input_b)
            loss = loss_fn(out, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        scheduler.step()

        # Evaluate the model
        train_acc = accuracy_cal(train_data)
        valid_acc = accuracy_cal(valid_data)

        if args.log_dir is not None:
            train_logger.add_scalar('accuracy', train_acc, global_step)
            train_logger.add_scalar('loss', loss, global_step)
            valid_logger.add_scalar('accuracy', valid_acc, global_step)

        if epoch % verb_step == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            print('Train accuracy: %f' % train_acc)
            print('Valid accuracy: %f' % valid_acc)

        if valid_acc > objective:
            objective = valid_acc
            objective_epoch = epoch
            save_model(model, namer(valid_acc))

        if epoch == n_epochs - 1:
            if objective_epoch is not None and n_epochs - objective_epoch < args.extension:
                n_epochs += args.extension

        epoch += 1
    # save_model(model)


if __name__ == '__main__':
    import argparse

    # Put custom arguments here
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

    parser.add_argument('-obj', '--objective', type=float, default=0.865)
    parser.add_argument('-ext', '--extension', type=int, default=10)
    parser.add_argument('-trains', '--n_trains', type=int, default=1)

    args = parser.parse_args()

    for _ in range(args.n_trains):
        train(args)