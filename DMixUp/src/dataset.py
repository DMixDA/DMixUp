from torchvision import datasets
import torchvision.transforms as transforms


def get_dataset(dataset_name, path='/database'):
    if dataset_name in ['art','clipart','product','real world']:  # office-home
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        tr_dataset = datasets.ImageFolder(path + '/officehomeD/' + dataset_name + '/', data_transforms['train'])
        te_dataset = datasets.ImageFolder(path + '/officehomeD/' + dataset_name + '/', data_transforms['test'])
        print('{} train set size: {}'.format(dataset_name, len(tr_dataset)))
        print('{} test set size: {}'.format(dataset_name, len(te_dataset)))

    else:
        raise ValueError('Dataset %s not found!' % dataset_name)
    return tr_dataset, te_dataset
