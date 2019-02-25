
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import utils,transforms

mnist_test = torchvision.datasets.MNIST(
    r'C:\rotation_face\PyTorch-Tutorial\tutorial-contents\mnist', train=False,
    download=True
)
print('testset:', len(mnist_test))

f = open('mnist_test.txt', 'w')
for i, (img, label) in enumerate(mnist_test):
    img_path = "./mnist_test/" + str(i) + ".jpg"
    img.save(img_path, 'jpeg')
    f.write(img_path + ' ' + str(label) + '\n')
f.close()
imgs = []


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1][7])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batchfrom dataloader')


if __name__ == "__main__":
    train_data = MyDataset(txt='mnist_test.txt', transform = transforms.ToTensor())
    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    print(len(data_loader))
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if (i < 4):
            print(i, batch_x.size(), batch_y.size())
            show_batch(batch_x)
            plt.axis('off')
            plt.show()

