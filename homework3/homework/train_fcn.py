import torch
import numpy as np
from torchvision import transforms

from .models import FCN, save_model, load_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

from datetime import datetime


def train(args):
    from os import path
    model = FCN()
    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    from os import path

    def accuracy_cal(data) -> float:
        confusion_matrix = ConfusionMatrix()
        for input_b, target_b in data:
            input_b = input_b.to(device)
            target_b = target_b.to(device).long()

            out_b = torch.argmax(model(input_b), dim=1)
            confusion_matrix.add(out_b, target_b)

        return confusion_matrix.global_accuracy, confusion_matrix.iou

    def namer(acc):
        st = str(acc) + '-' + datetime.now().strftime('%d-%m-%Y,%H-%M-%S')
        return st

    def loss_weigths(weigths):
        normed_weights = [1 - (x / sum(weigths)) for x in weigths]
        normed_weights = torch.FloatTensor(normed_weights)
        return normed_weights  
    

    print('start training')
    # The device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # The model
    if not args.load:
        model = FCN().to(device)
    else:
        model = load_model('fcn').to(device)

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
    train_data = load_dense_data('dense_data/train', batch_size=bs, num_workers=4,
                                 transform=dense_transforms.Compose(
                                     (
                                         dense_transforms.ToTensor(),
                                         dense_transforms.ColorJitter(0.8, 0.8, 0.8, 0.4),
                                         dense_transforms.RandomHorizontalFlip()
                                     )
                                 ))
    valid_data = load_dense_data('dense_data/valid', batch_size=bs)

    # The loss fn
    loss_fn = torch.nn.CrossEntropyLoss(loss_weigths(DENSE_CLASS_DISTRIBUTION)).to(device)

    # The optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Accuracy series
    global_step = 0
    epoch = 0
    while epoch < n_epochs:
        confusion_matrix = ConfusionMatrix()
        # Shuffle batches from the data loader
        for input_b, target_b in train_data:
            # Compute the out and the loss
            input_b = input_b.to(device)
            target_b = target_b.to(device).long()

            out = model(input_b)
            confusion_matrix.add(out.argmax(1), target_b)

            loss = loss_fn(out, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log_dir is not None and global_step % 100 == 0:
                train_logger.add_scalar('loss', loss, global_step)

            global_step += 1

        scheduler.step()

        # Evaluate the model
        train_acc, train_iou = confusion_matrix.global_accuracy, confusion_matrix.iou
        valid_acc, valid_iou = accuracy_cal(valid_data)

        # Epoch logs
        if args.log_dir is not None:
            train_logger.add_scalar('accuracy', train_acc, global_step)
            train_logger.add_scalar('iou', train_iou, global_step)

            valid_logger.add_scalar('accuracy', valid_acc, global_step)
            valid_logger.add_scalar('iou', valid_iou, global_step)

        # Epoch prints
        if epoch % verb_step == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            print('Train accuracy: %0.4f' % train_acc)
            print('Train iou: %0.4f' % train_iou)
            
            print('Valid accuracy: %0.4f' % valid_acc)
            print('Valid iou: %0.4f' % valid_iou)
        
        # Epoch save
        if valid_acc > objective:
            objective = valid_acc
            objective_epoch = epoch
            save_model(model, namer(valid_acc))

        # Epoch extension
        if epoch == n_epochs - 1:
            if objective_epoch is not None and n_epochs - objective_epoch < args.extension:
                n_epochs += args.extension

        epoch += 1
        
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

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
