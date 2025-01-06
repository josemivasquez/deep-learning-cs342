import torch
from .utils import load_data, load_all
from . import dense_transforms
from .models import Detector, save_model, load_model
import torch.utils.tensorboard as tb
from os import path

import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logging', help="Data directory")
    parser.add_argument('-n', '--epochs', default=10, type=int, help="Epochs to run SGD")
    parser.add_argument('--data_path_train', default='./data/train', help="Data directory")
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate")
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-o', '--optimizer', default='optim.Adam(parameters)')
    parser.add_argument('-sl', '--schedule_lr', action='store_false')

    args = parser.parse_args()

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = dense_transforms.Compose([dense_transforms.ToTensor()])

    all_data = load_all('./hockey_training_data/')
    
    train = all_data[:int(len(all_data) * 0.8)]
    valid = all_data[int(len(all_data) * 0.8):]
    
    train_data = load_data(train, transform=transform, num_workers=4, batch_size=args.batch_size)
    valid_data = load_data(valid, transform=transform, num_workers=4, batch_size=args.batch_size)

    model = Detector().to(device)

    optimizer_map = {
      'sgd' : torch.optim.SGD(
          model.parameters(), lr=args.lr, 
          momentum=0.9, weight_decay=4e-03
      ),

      'adam' : torch.optim.Adam(model.parameters(), lr=args.lr)
    }

    optimizer = optimizer_map['adam']
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.1)

    print_each = 1
    global_step = 0
    print('start train')
    for epoch in range(args.epochs):
        it = 0
        losses = []
        for img, label in train_data:
            img = img.to(device)
            label = label.to(device)
            pred = model(img)

            # label 1 -> sigm(-pred) = 1 - sigm(pred)
            # label 0 -> sigm(pred) = p(y = 1)
            gamma = 2
            p_no_l = torch.sigmoid(pred * (1 - 2*label))
            loss = (bce_loss(pred, label) * (p_no_l**gamma)).mean() / p_no_l.mean()

            losses.append(float(loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            it += 1
        
        loss_mean = sum(losses) / len(losses)

        if epoch % print_each == 0:
            print('Epoch: ', epoch)
            print('Loss mean: ', loss_mean)
            train_logger.add_scalar('loss', loss_mean, global_step)
        
        scheduler.step()
        global_step += 1
    
    print('train finished')
    save_model(model)

