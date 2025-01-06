from PIL import Image
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.data = []

        labels_path = dataset_path + '/labels.csv'
        labels_reader = csv.reader(open(labels_path))

        transform = transforms.Compose([transforms.ToTensor()])

        header = True
        for row in labels_reader:
            if header:
                header = False
                continue

            img_file_name = row[0]
            category = LABEL_NAMES.index(row[1])

            img_path = dataset_path + '/' + img_file_name
            img_tensor = transform(Image.open(img_path))

            self.data.append((img_tensor, category))

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
