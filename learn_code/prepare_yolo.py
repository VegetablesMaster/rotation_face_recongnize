import cv2
import os
import random
import shutil
import time
import argparse
from functools import reduce
from learn_code.prepare_data import get_rect
from jinja2 import Environment, FileSystemLoader

# TODO:: 在这里配置你编译生成的darknet.exe的环境目录 和准备的训练素材所在文件夹的位置
yolo_train_enviroment_path = r'C:\Users\vegetable master\Desktop\yolo_cuda\darknet\build\darknet\x64'
prepare_data_path = os.getcwd()                                 # 我的素材就放当前目录的\\video_hub中


def yolo_pic_process():                                         # 一个自己给数据打标的垃圾程序
    """
    请勿使用！！！ 与官方配套的打标程序不兼容！！！
    好不容易写的，不想删掉 ~_~"
    :return:
    """
    count = 0
    pic_dir = prepare_data_path + '\\video_hub\\app_photo'
    marked_dir = prepare_data_path + '\\video_hub\\app_text'          # 打标数据的存放处
    for root, dirs, files in os.walk(pic_dir):                  # 递归主文件下子文件夹下的所有图片
        for dir in dirs:
            for sub_root, sub_dirs, sub_files in os.walk(os.path.join(root, dir)):
                for file in sub_files:
                    pic_file_name = os.path.splitext(file)[0]   # pic 与 text 同名，后缀不同
                    pic_file_path = pic_dir + '\\' + dir + '\\' + file
                    text_file_path = marked_dir + '\\' + pic_file_name+'.txt'  # 绝对路径
                    if not os.path.exists(text_file_path):
                        img = cv2.imread(pic_file_path)
                        x, y, z = img.shape
                        tl, br = get_rect(img)
                        yolo_class_id = 0                       # 类别id 默认为0
                        yolo_x = float(tl[0] / x)               # 归一化的x坐标
                        yolo_y = float(tl[1] / y)               # 归一化的x坐标
                        yolo_w = float(abs(tl[0] - br[0]) / x)  # 归一化的宽度w
                        yolo_h = float(abs(tl[1] - br[1]) / y)  # 归一化的高度h
                        tmp_str = str(yolo_class_id) + " "
                        tmp_str = tmp_str + str(yolo_x) + " "
                        tmp_str = tmp_str + str(yolo_y) + " "
                        tmp_str = tmp_str + str(yolo_w) + " "
                        tmp_str = tmp_str + str(yolo_h)
                        with open(text_file_path, 'w') as f:    # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                            f.write(tmp_str)
    print(count)


def yolo_change_mark():                                         # 垃圾代码,提供了一些可以借鉴的代码片段吧
    """
    试图做一下挣扎，转换了一部分自己打标的数据与官方的数据。 但效果巨差！！！，有部分高级编程的语法可以借鉴。
    :return:
    """
    start_time = time.time()
    marked_dir = prepare_data_path + '\\video_hub\\app_text'          # 打标数据的存放处
    train_dir = prepare_data_path + '\\video_hub\\train_yolo'
    for root, dirs, files in os.walk(marked_dir):               # 递归主文件下子文件夹下的所有图片
        total_file_num = len(files)
        for file in files:
            file_index = files.index(file)
            if file_index % 100 == 0:
                time_pass = str(round(time.time() - start_time, 1))
                finish_percent = str(round(float(file_index/total_file_num), 3))
                print('process finished: ' + finish_percent + '   used time: ' + time_pass + ' s')
            with open(marked_dir + '\\' + file, 'r') as f:
                tmp_str = f.readline()
            temp_dic = tmp_str.split(' ')                                # 按照空格分割文本
            tep_num = [round(float(num), 6) for num in temp_dic[1:5]]    # 将str转换为6位浮点数
            tep_num[0] = tep_num[0] - 0.8 * tep_num[0] * tep_num[2]
            tep_num[1] = tep_num[1] + 5 * tep_num[1] * tep_num[3]
            if tep_num[0] <= 0.2:
                continue
            if tep_num[1] <= 0.25:
                continue
            elif tep_num[1] >= 0.6:
                tep_num[1] = tep_num[1] * 0.9
            tep_num[2] = tep_num[2] + 0.05
            tep_num[3] = tep_num[3] + 0.1
            tep_num = [round(num, 6) for num in tep_num]
            tmp_str_n = reduce(lambda x, y: str(x) + ' ' + str(y), tep_num)  # 浮点数转换成字符串
            tmp_str_n = str(temp_dic[0]) + ' ' + tmp_str_n
            with open(file, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                f.write(tmp_str_n)      # 测试 f.write(tmp_str + '\n' +tmp_str_n)
            targetfile = train_dir + '\\' + file
            if os.path.isfile(targetfile):
                os.remove(targetfile)
            shutil.move(file, train_dir)  # 将配置文件剪切到yolo所在的环境目录下
    yolo_pic_mark()


def yolo_pic_examine():                                               # 检测打标信息的正确性
    """
    配套垃圾程序！！！ 可以看下打标的效果。 孤芳自赏一下.
    :return:
    """
    marked_dir = prepare_data_path + '\\video_hub\\app_text'          # 打标数据的存放处
    pic_dir = prepare_data_path + '\\video_hub\\app_photo\\all_in'    # 打标数据对应的图片
    for root, dirs, files in os.walk(marked_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                text_file_path = marked_dir + '\\' + file
                pic_file_path = pic_dir + '\\' + os.path.splitext(file)[0] + '.jpg'
                if not os.path.isfile(pic_file_path):                 # 如果找不到对应打标文件的图片，不记录该训练信息
                    print('无法找到该打标信息对应的图片: '+str(file))
                    continue
                img = cv2.imread(pic_file_path)
                x, y, z = img.shape
                with open(text_file_path, 'r') as f:                  # 读取打标信息并绘制方框
                    temp_str = f.readline()
                temp_dic = temp_str.split(' ')
                tl = (int(x*float(temp_dic[1])), int(y*float(temp_dic[2])))
                br = (int(tl[0] + x*float(temp_dic[3])), int(tl[1] + y*float(temp_dic[4])))
                cv2.rectangle(img, tl, br, (255, 0, 0))
                cv2.imshow(str(os.path.splitext(file)[0]), img)
                if cv2.waitKey(10000) & 0xFF == ord('c'):             # 如果合格按c继续下一张，否则删除该达标信息
                    cv2.destroyAllWindows()
                    continue
                else:
                    os.remove(pic_file_path)


def yolo_pic_mark():
    """
    一个调用官方打标程序的命令，官方打标命令需要编译安装，别担心，比darknet好装很多 TODO:: 配置训练素材(图片)和类名的文件的路径
    :return:
    """
    img_path = prepare_data_path + '\\video_hub\\train_yolo'
    train_text_path = yolo_train_enviroment_path + '\\my_data\\train.txt'
    obj_name_file = yolo_train_enviroment_path + '\\' + 'my_own_obj.names'
    cmd_str = 'yolo_mark.exe ' + img_path + ' ' + train_text_path + ' ' + obj_name_file
    print(cmd_str)


def yolo_train_change_sample():
    """
    生成新的训练集测试集的样本信息针对不同的素材，每次分出来的训练集和测试集是不同的 TODO:: 把打标数据和图片跟别放到指定的位置
    :return:
    """
    start_time = time.time()
    pic_dir = prepare_data_path + '\\video_hub\\train_yolo'
    marked_dir = prepare_data_path + '\\video_hub\\yolo_finish'     # 打标数据的存放处
    train_txt_path = prepare_data_path + '\\train.txt'              # 记录训练集和样测试集图片信息的文本
    valid_txt_path = prepare_data_path + '\\valid.txt'
    f_train = open(train_txt_path, 'w')
    f_valid = open(valid_txt_path, 'w')
    for root, dirs, files in os.walk(marked_dir):
        total_file_num = len(files)
        for file in files:
            file_index = files.index(file)
            if file_index % 100 == 0:                               # 其实可以拉出去封装一个函数，做成迭代器
                time_pass = str(round(time.time() - start_time, 1))
                finish_percent = str(round(float(file_index/total_file_num), 3))
                print('process finished: ' + finish_percent + '   used time: ' + time_pass + ' s')
            targetfile = pic_dir + '\\' + file                      # 删除原始的打标数据
            if os.path.isfile(targetfile):
                os.remove(targetfile)
            pic_file_name = os.path.splitext(file)[0]
            pic_file_path = pic_dir + '\\' + pic_file_name + '.jpg'
            if not os.path.isfile(pic_file_path):                   # 如果找不到对应打标文件的图片，不记录该训练信息
                continue
            shutil.copy(root + '\\' + file, pic_dir)                # 复制新的打标数据到和图片相同的文件夹下
            if random.random() > 0:                                 # 采用100%的训练数据
                if random.random() > 0.1:                           # 训练集为90%
                    f_train.write(pic_file_path+'\n')               # 分别记录训练信息到训练集和测试集中
                else:
                    f_valid.write(pic_file_path+'\n')
    f_train.close()
    f_valid.close()
    targetfile = yolo_train_enviroment_path + '\\my_data\\train.txt'
    if os.path.isfile(targetfile):                                  # 删除原来的配置文件
        os.remove(targetfile)
    targetfile = yolo_train_enviroment_path + '\\my_data\\valid.txt'
    if os.path.isfile(targetfile):
        os.remove(targetfile)
    shutil.move(train_txt_path, yolo_train_enviroment_path + '\\my_data')  # 将配置文件剪切到yolo所在的环境目录下
    shutil.move(valid_txt_path, yolo_train_enviroment_path + '\\my_data')


def yolo_train_cmd(class_num=1):
    """
    配置darknet的训练环境，并生成其训练命令 TODO:: 有兴趣的同学可以找到配置模板学习参数是如何配置的
    :param class_num:你要检测的目标有几类，我们这里当然只识别脸而已
    :return: 返回训练命令，该命令请在darknet.exe所在的目录下通过cmd运行，不推荐在pycharm自带terminal环境内运行
    """
    obj_name_file = 'my_own_obj.names'                      # 目标检测中的类名文件
    f = open(obj_name_file, 'w')
    f.write('face\n')                                       # 目前默认一个类
    f.close()
    obj_data_file = 'my_own_obj.data'                       # 记录训练图片的所在位置
    f = open(obj_data_file, 'w')
    f.write('classes = ' + str(class_num) + ' \n')          # 类的数目
    f.write('train = my_data\\train.txt\n')                 # 存储训练集图片路径的文件位置
    f.write('valid = my_data\\valid.txt\n')                 # 存储验证集图片路径的文件位置
    f.write('names = my_own_obj.names\n')                   # obj_name_file 的存放位置
    f.write('backup = backup\\\n')
    f.close()
    cfg_file = 'my_own_yolov3.cfg'                          # 训练的配置文件(超参数和网络模型)
    env = Environment(loader=FileSystemLoader(yolo_train_enviroment_path))
    tpl = env.get_template('yolov3_jinja_model.cfg')        # 用jinjia模板引擎生成cfg配置文件
    with open(cfg_file, 'w') as f:
        render_content = tpl.render(classes=class_num, filters=(class_num+5)*3)
        f.write(render_content)                             # 根据class的数目对模板进行渲染
    targetfile = yolo_train_enviroment_path + '\\' + cfg_file
    if os.path.isfile(targetfile):                          # 删除原来的配置文件
        os.remove(targetfile)
    targetfile = yolo_train_enviroment_path + '\\' + obj_data_file
    if os.path.isfile(targetfile):
        os.remove(targetfile)
    targetfile = yolo_train_enviroment_path + '\\' + obj_name_file
    if os.path.isfile(targetfile):
        os.remove(targetfile)
    shutil.move(cfg_file, yolo_train_enviroment_path)       # 把生成的配置文件移动到yolo环境下
    shutil.move(obj_data_file, yolo_train_enviroment_path)
    shutil.move(obj_name_file, yolo_train_enviroment_path)
    cmd_str = 'darknet.exe detector train '
    cmd_str = cmd_str + obj_data_file + ' '                 # obj_data_file文件的路径
    cmd_str = cmd_str + cfg_file + ' '                      # cfg文件的路径
    cmd_str = cmd_str + 'darknet53.conv.74'                 # 使用的预模型路径
    print(cmd_str)                                          # 该命令在yolo所在的环境目录下运行


def yolo_test(test_file=None):
    """
    一个快速帮助你测试yolo的小程序，默认情况下只产生测试命令 TODO::可以通过设置文件名直接运行你想要的测试
    :param test_file: 这些文件需要在‘my_data\\'目录下，摄像头直接输入'cap0'等。
    :return:
    """
    global test_choose_flag
    pic_path = 'my_data\\test1.jpg'
    video_path = 'my_data\\jiaqi.mp4'
    cap_path = '-c 0'
    if isinstance(test_file, str):
        file_type = test_file.split('.')[-1]
        if file_type == 'mp4':
            video_path = 'my_data\\'+test_file
            test_choose_flag = 'video'
        elif file_type == 'jpg':
            pic_path = 'my_data\\'+test_file
            test_choose_flag = 'pic'
        elif file_type[:-1] == 'cap':
            cap_path = '-c ' + file_type[-1]
            test_choose_flag = 'cap'
    else:
        test_choose_flag = None

    obj_data_file = 'my_own_obj.data'                       # 记录训练图片的所在位置
    cfg_file = 'my_own_yolov3.cfg'                          # 训练的配置文件(超参数和网络模型)
    wight_path = 'backup\\my_own_yolov3_5000.weights'       # 模型参数
    # 这个地方可以设置置信度 啥是置信度 %_%->@
    config_path = ' {} {} {} -i 0 -thresh 0.2 -ext_output '.format(obj_data_file, cfg_file, wight_path)
    cmd_str = 'darknet.exe detector test '                  # 测试图片
    jpg_cmd_str = cmd_str + config_path + pic_path
    print('test_pic: {}\n'.format(jpg_cmd_str))

    cmd_str = 'darknet.exe detector demo '                  # 测试视频
    video_cmd_str = cmd_str + config_path + video_path
    print('test_video: {}\n'.format(video_cmd_str))

    cmd_str = 'darknet.exe detector demo '                  # 测试摄像头
    cap_cmd_str = cmd_str + config_path + cap_path
    print('test_cap: {}\n'.format(cap_cmd_str))
    os.system('chcp 65001')
    os.chdir(yolo_train_enviroment_path)
    if not test_choose_flag != 'pic':
        os.system(jpg_cmd_str)
    elif test_choose_flag == 'video':
        os.system(video_cmd_str)
    elif test_choose_flag == 'cap':
        os.system(cap_cmd_str)
    else:
        print('请在darknet.exe所在的目录下运行')


def main():
    cwd = os.path.dirname(__file__)
    a = os.path.split(__file__)
    parser = argparse.ArgumentParser()
    # parser.add_argument("square", help="display a square of a given number",
    #                     type=int)
    parser.add_argument('-d', "--demo", help="一个快速帮助你测试yolo的小程序，默认情况下只产生测试命令", action="store_true")
    parser.add_argument('-t', "--train", help="配置darknet的训练环境，并生成其训练命令", action="store_true")
    parser.add_argument('-cs', "--change_sample", help="生成新的训练集测试集的样本信息针对不同的素材，每次分出来的训练集和测试集\
        是不同的", action="store_true")
    parser.add_argument('-f', '--filename', help="可以通过设置文件名直接运行你想要的测试")
    args = parser.parse_args()
    if args.demo:
        if args.filename:
            yolo_test(args.filename)
        else:
            yolo_test()
    elif args.train:
        yolo_train_cmd()
    elif args.change_sample:
        yolo_train_change_sample()
    else:
        example = 'python prepare_yolo.py -demo -filename leyi.mp4'
        print("you can use cmd like : {} \nor run in cmd and use -h to see help".format(example))


if __name__ == '__main__':
    yolo_test('leyi.mp4')
    # main()
    """
    usage: prepare_yolo.py [-h] [-d] [-t] [-cs] [-f FILENAME]

    optional arguments:
    -h, --help            show this help message and exit
    -d, --demo            一个快速帮助你测试yolo的小程序，默认情况下只产生测试命令
    -t, --train           配置darknet的训练环境，并生成其训练命令
    -cs, --change_sample  生成新的训练集测试集的样本信息针对不同的素材，每次分出来的训练集和测试集 是不同的
    -f FILENAME, --filename FILENAME   可以通过设置文件名直接运行你想要的测试                           
    """
