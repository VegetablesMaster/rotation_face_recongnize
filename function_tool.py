import os
import shutil


def rewrite_file(filepath, targetpath):
    """
    删除旧配置将新的配置文件copy到配置文件夹
    :param filepath: 最好是绝对路径
    :param targetpath: 目标文件夹
    :return: 新文件的绝对路径
    """
    _, filename = os.path.split(filepath)
    targetfile = os.path.join(targetpath, filename)
    if os.path.exists(targetfile):
        os.remove(targetfile)
    shutil.move(filepath, targetpath)
    return os.path.join(targetpath, filename)


