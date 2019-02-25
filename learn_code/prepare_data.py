import glob
import os
import shutil
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms
from torchvision.transforms import ToTensor

from learn_code.darknet import configDetect
from function_tool import rewrite_file

resize_x = 50
resize_y = 50


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderSplitter:
    # images should be placed in folders like:
    # --root
    # ----root\dogs
    # ----root\dogs\image1.png
    # ----root\dogs\image2.png
    # ----root\cats
    # ----root\cats\image1.png
    # ----root\cats\image2.png
    # path: the root of the image folder
    def __init__(self, path, train_size=0.8):
        self.path = path
        self.train_size = train_size
        self.class2num = {}
        self.num2class = {}
        self.class_nums = {}
        self.data_x_path = []
        self.data_y_label = []
        self.x_train = []
        self.x_valid = []
        self.y_train = []
        self.y_valid = []
        for root, dirs, files in os.walk(path):
            if len(files) == 0 and len(dirs) > 1:
                for i, dir1 in enumerate(dirs):
                    self.num2class[i] = dir1
                    self.class2num[dir1] = i
            elif len(files) > 1 and len(dirs) == 0:
                category = ""
                for key in self.class2num.keys():
                    if key in root:
                        category = key
                        break
                label = self.class2num[category]
                self.class_nums[label] = 0
                for file1 in files:
                    self.data_x_path.append(os.path.join(root, file1))
                    self.data_y_label.append(label)
                    self.class_nums[label] += 1
            else:
                raise RuntimeError("please check the folder structure!")
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_path, self.data_y_label,
                                                                                  shuffle=True,
                                                                                  train_size=self.train_size)

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_valid, self.y_valid


class DatasetFromFilename(Dataset):
    # x: a list of image file full path
    # y: a list of image categories
    def __init__(self, x, y, transforms=None):
        super(DatasetFromFilename, self).__init__()
        self.x = x
        self.y = y
        if transforms == None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx])
        img = img.convert("RGB")
        return self.transforms(img), torch.tensor([[self.y[idx]]])


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader, train_size=0.9):
        fh = open(txt, 'r')
        imgs = []
        label = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(' ')
            imgs.append(words[0])
            label.append(int(words[1]))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(imgs, label,
                                                                                  shuffle=True,
                                                                                  train_size=train_size)
        fh.close()

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_valid, self.y_valid

    def get_val_Date(self):
        imgs = torch.Tensor(1, 3, 100, 100)
        for fn in self.x_valid:
            img = torch.unsqueeze(self.transform(self.loader(fn)), dim=0)
            imgs = torch.cat((imgs, img), 0)
        label = torch.Tensor(self.y_valid)
        return imgs[1:], label, self.x_valid

    def __getitem__(self, index):
        fn = self.x_train[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batchfrom dataloader')


def get_rect(im, title="右取消".encode("gbk").decode(errors="ignore")):  # cv2默认gbk编码，python为utf-8
    """
    一个通过CV鼠标选择图像区域的程序的！ 不是我写的，但经过了适当的修改，是个理解CV2鼠标响应不错的例子
    :param im: 图像名称
    :param title: 窗口名称
    :return: 选择区域的坐标(tl, br) tl=(y1,x1), br=(y2,x2)
    """
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
                    'released_once': False}
    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):  # 每当鼠标移动或点击时，都会触发这个函数
        param['current_pos'] = (x, y)
        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True
        if (event == cv2.EVENT_RBUTTONDOWN):
            param['tl'] = None
        if (event == cv2.EVENT_LBUTTONDOWN):
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    try:
        cv2.imshow(title, im)
    except:
        pass
    while mouse_params['br'] is None:
        im_draw = np.copy(im)
        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                          mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)
    cv2.destroyWindow(title)
    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
          min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
          max(mouse_params['tl'][1], mouse_params['br'][1]))
    return (tl, br)


def video_process(postive_flag=True):  # 直接从视频中截取适当的训练素材
    """
    从视频中直接选取敏感区域并保存到素材库中，本项目的第一个段代码，直接了当，无需多言。
    PS !!! &#_#``` 不同视频会根据名称自己保存到对应的文件夹中,TODO ：：但需要提前准备好文件夹，你可修改代码自动实现这个功能，是的，我懒！
    :param postive_flag: 选择是否是正面样本，如果是负面样本就换个区域放
    :return:
    """
    video_pwd = os.getcwd() + '\\video_hub'
    L = []
    print(video_pwd)
    for root, dirs, files in os.walk(video_pwd):  # 把所有的.mp4文件都先挑出来，稍后就一起收拾它们！
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, file))

    for video in L:
        cap = cv2.VideoCapture(video)
        if postive_flag:  # 每一个正面素材都会属于一个具体的类名，不妨假设为视频名
            output_dir = os.getcwd() + '\\video_hub\\train_face\\' + os.path.splitext(os.path.split(video)[1])[0]
        else:
            output_dir = os.getcwd() + '\\video_hub\\train_face\\negative'  # 既然大家都是反面素材了，那就没必要按视频名称分类了
        count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            flag = cv2.waitKey(10000)
            if flag == 27:
                break
            elif flag == ord('c'):
                continue
            (a, b) = get_rect(frame, title='get_rect')  # 这个程序真的不是pull&run，至少真的修改了下
            cv2.imwrite(output_dir + '\\' + str(count) + '.jpg', frame[a[1]:b[1], a[0]:b[0]])
            count = count + 1
        cap.release()  # 为了严谨，其实运行完了它会自动释放的


def yolo_video_process():
    """
    先通过yolo检测图像中是否有检测目标的信息，将目标信息分类后保存在文件夹中。
    :return:
    """
    video_pwd = os.getcwd() + '\\video_hub\\app_video_0221'
    output_dir = os.getcwd() + '\\video_hub\\app_video_0221_output'
    L = []
    for root, dirs, files in os.walk(video_pwd):  # 把所有的.mp4文件都先挑出来，稍后就一起收拾它们！
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, file))
    for video in L:
        cap = cv2.VideoCapture(video)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            cv2.imwrite('temp.jpg', frame)
            result = configDetect('temp.jpg')                   # TODO :: 提前在darknet.py中配置好yolo所在的环境
            # print(result)c
            img = cv2.imread('temp.jpg')
            try:                                                                            # 有时候检测不到人脸
                b = [int(num) for num in result[0][2]]
            except IndexError:
                continue
            tr_point = (int(b[0] - 0.5 * b[2]), int(b[1] - 0.5 * b[3]))                     # 在cv2中框出人脸
            lb_point = [int(b[0] + 0.5 * b[2]), int(b[1] + 0.5 * b[3])]
            tr_point = list(map(lambda x: x if (x > 0) else 0, tr_point))                   # map 结果需要list
            lb_point[0] = lb_point[0] if (lb_point[0] < img.shape[1]) else img.shape[1]     # 防止越界
            lb_point[1] = lb_point[1] if (lb_point[1] < img.shape[0]) else img.shape[0]
            tr = (tr_point[0], tr_point[1])
            lb = (lb_point[0], lb_point[1])
            cv2.rectangle(img, tr, lb, (255, 0, 0))
            cv2.imshow('show', img)
            output_img = img[tr[1]:lb[1], tr[0]:lb[0]]
            cv2.imshow('show_out', output_img)
            flag = cv2.waitKey(10000)
            if flag in [ord(str(x)) for x in [1, 2, 3, 6, 9, 8, 7, 4]]:
                label_dic = {'49': '-45', '50': '0', '51': '45', '52': '-90', '54': '90',
                             '55': '-135', '56': '180', '57': '135'}
                output_path = '{}\\{}'.format(output_dir, label_dic[str(flag)])
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                output_img_filename = output_path+'\\'+str(time.time())+'.jpg'              # 记着后缀.jpg
                cv2.imwrite(output_img_filename, output_img)
            elif flag == ord('b'):
                break
            else:
                continue
        cap.release()  # 为了严谨，其实运行完了它会自动释放的


def photo_process():
    photo_hub_pwd = os.getcwd() + '\\video_hub\\train_face'
    for root, dirs, files in os.walk(photo_hub_pwd):
        for dir in dirs:
            L = []
            filenames = os.listdir(photo_hub_pwd + '\\' + dir)
            filenames.sort(key=lambda x: int(x[:-4]))
            for file in filenames:
                if os.path.splitext(file)[1] == '.jpg':
                    L.append(os.path.join(root + '\\' + dir, file))
            count = 0
            if dir != 'negative':
                output_dir = os.getcwd() + '\\video_hub\\train_face_finish\\positive\\' + dir
            else:
                output_dir = os.getcwd() + '\\video_hub\\train_face_finish\\' + dir
            shutil.rmtree(output_dir)
            for photo in L:
                img = cv2.imread(photo)
                if dir != 'negative':
                    img = cv2.resize(img, (resize_x, resize_y))
                else:
                    img = cv2.resize(img, (resize_x + 100, resize_y + 100))
                img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
                img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
                img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
                cv2.imwrite(output_dir + '\\' + str(count) + '.jpg', img)
                count = count + 1
                print(photo)


def haar_train(hs_num=15, h_mhr=0.999, h_mfar=0.5, mode='ALL'):
    s_count = 0
    b_count = 0
    train_cmd = ['', '', '']
    photo_hub_pwd = os.getcwd() + '\\video_hub\\train_face_finish\\postive'
    postive_file_name = os.getcwd() + "\\ht_conf.txt"
    haar_train_img_conf = open(postive_file_name, "w")
    for root, dirs, files in os.walk(photo_hub_pwd):
        for dir in dirs:
            filenames = os.listdir(photo_hub_pwd + '\\' + dir)
            filenames.sort(key=lambda x: int(x[:-4]))
            for file in filenames:
                if os.path.splitext(file)[1] == '.jpg':
                    s_count = s_count + 1
                    out_str = os.path.join('video_hub\\train_face_finish\\postive\\' + dir, file) + ' 1 0 0 ' + str(
                        resize_x) + ' ' + str(resize_y)
                    print(out_str)
                    haar_train_img_conf.write(out_str + "\n")
    train_cmd[0] = 'opencv_createsamples.exe -info ' + postive_file_name + ' -vec samples.vec -num ' + str(
        s_count) + ' -w ' + str(resize_x) + ' -h ' + str(resize_y)
    photo_hub_pwd = os.getcwd() + '\\video_hub\\train_face_finish\\negative'
    negative_file_name = os.getcwd() + "\\ht_bg_conf.txt"
    haar_train_img_conf = open(negative_file_name, "w")
    filenames = os.listdir(photo_hub_pwd)
    filenames.sort(key=lambda x: int(x[:-4]))
    for file in filenames:
        if os.path.splitext(file)[1] == '.jpg':
            b_count = b_count + 1
            out_str = os.path.join(os.getcwd() + '\\video_hub\\train_face_finish\\negative\\', file)
            print(out_str)
            haar_train_img_conf.write(out_str + "\n")
    train_cmd[1] = 'echo:training'
    train_cmd[2] = 'opencv_traincascade.exe -data out'  # 指定保存训练结果的文件夹；
    train_cmd[2] = train_cmd[2] + ' -vec samples.vec'  # 指定正样本集；
    train_cmd[2] = train_cmd[2] + ' -bg ht_bg_conf.txt'  # 指定负样本的描述文件夹；
    train_cmd[2] = train_cmd[2] + ' -numPos ' + str(s_count)  # 指定每一级参与训练的正样本的数目（要小于正样本总数）；
    train_cmd[2] = train_cmd[2] + ' -numNeg ' + str(b_count)  # 指定每一级参与训练的负样本的数目（可以大于负样本图片的总数）；
    train_cmd[2] = train_cmd[2] + ' -numStage ' + str(hs_num)  # 训练的级数；
    train_cmd[2] = train_cmd[2] + ' -w ' + str(resize_x)  # 正样本的宽；
    train_cmd[2] = train_cmd[2] + ' -h ' + str(resize_y)  # 正样本的高；
    train_cmd[2] = train_cmd[2] + ' -minHitRate ' + str(h_mhr)  # 每一级需要达到的命中率（一般取值0.95 - 0.995）
    train_cmd[2] = train_cmd[2] + ' -maxFalseAlarmRate ' + str(h_mfar)  # 每一级所允许的最大误检率
    train_cmd[2] = train_cmd[2] + ' -mode: ' + mode  # 使用Haar - like特征时使用，可选BASIC、CORE或者ALL
    for cmd in train_cmd:
        print(cmd)
        # os.system(cmd)


def face_prepare(test_flag=False):
    photo_hub_pwd = os.getcwd() + '\\video_hub\\train_face_finish\\postive'
    file_name = os.getcwd() + "\\face_recognize.txt"
    conf = open(file_name, "w")
    for root, dirs, files in os.walk(photo_hub_pwd):
        for dir in dirs:
            filenames = os.listdir(photo_hub_pwd + '\\' + dir)
            filenames.sort(key=lambda x: int(x[:-4]))  # 图片名称的数字大小排序
            for file in filenames:
                if os.path.splitext(file)[1] == '.jpg':
                    out_str = os.path.join('video_hub\\train_face_finish\\postive\\' + dir, file) + ' ' \
                              + str(int(dirs.index(dir) / 2))  # 一个人有两种情况
                    print(out_str)
                    conf.write(out_str + "\n")
    conf.close()
    if test_flag:
        train_data = MyDataset(txt=file_name, transform=transforms.ToTensor())
        data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
        print(len(data_loader))
        for i, (batch_x, batch_y) in enumerate(data_loader):
            if (i < 4):
                print(i, batch_x.size(), batch_y)
                show_batch(batch_x)
                plt.axis('off')
                plt.show()


def angle_prepare(test_flag=False):
    photo_hub_pwd = os.getcwd() + '\\video_hub\\app_video_0221_output'
    output_dir = os.getcwd() + '\\video_hub\\app_video_0221_output_equalize'
    conf_file_name = "\\face_angle.txt"
    resize_shap = [100, 100]
    conf_file = classify_conf_prepare(photo_hub_pwd, output_dir, conf_file_name, resize_shap)
    train_data = MyDataset(txt=conf_file, transform=transforms.ToTensor())
    if test_flag:
        data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
        print(len(data_loader))
        for i, (batch_x, batch_y) in enumerate(data_loader):
            if 4 > i:
                print(i, batch_x.size(), batch_y)
                show_batch(batch_x)
                plt.axis('off')
                plt.show()


def classify_conf_prepare(photo_hub_pwd, output_dir, conf_file_name, resize_shape):
    """
    处理待分类的图片到统一格式并生成分类的配置文件
    :param photo_hub_pwd: 素材位置
    :param output_dir: 处理好的素材
    :param conf_file_name: 统一生成到conf_hub中
    :param resize_shape: [x,y]
    :return: 配置文件的绝对路径
    """
    conf_file_path = os.path.split(__file__)[0] + '\\' + conf_file_name
    conf = open(conf_file_path, "w")                                                # 覆盖写

    for root, dirs, files in os.walk(photo_hub_pwd):
        for dir in dirs:
            L = []
            filenames = os.listdir(photo_hub_pwd + '\\' + dir)
            for file in filenames:
                if os.path.splitext(file)[1] == '.jpg':
                    L.append(os.path.join(root + '\\' + dir, file))
            count = 0
            output_path = output_dir + '\\' + dir
            if os.path.exists(output_path):                                 # 清空文件夹
                shutil.rmtree(output_path)
            os.mkdir(output_path)
            for photo in L:
                img = cv2.imread(photo)
                img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
                img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
                img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
                img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
                cv2.imwrite(output_path + '\\' + str(count) + '.jpg', img)
                out_str = output_path + '\\' + str(count) + '.jpg ' + str(dir)
                conf.write(out_str + "\n")
                count = count + 1
                print(out_str)
    conf.close()
    conf_file = rewrite_file(conf_file_path, os.getcwd() + '\\conf_hub')
    return conf_file


def haar_family_part():
    # video_process(False)
    photo_process()
    haar_train()


if __name__ == '__main__':
    angle_prepare(True)
