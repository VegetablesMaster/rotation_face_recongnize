import os
'''获得当前路径
'''
cwd=os.getcwd()
print(cwd)


'''
得到当前文件夹下的所有文件和文件夹
'''
print(os.listdir())


'''
delete file
'''
os.remove('sw724.vaps')
print(os.listdir())


'''
删除单个目录和多个目录
'''
os.removedir()
os.removedir()


'''
检查是否是文件／文件夹
'''
print(os.path.isfile('/Users/liuxiaolong/PycharmProjects/untitled/sw724.vaps'))
print(os.path.isdir('/Users/liuxiaolong/PycharmProjects/untitled/sw724.vaps'))


'''
检查文件路径是否存在
'''

print(os.path.exists('/Users/liuxiaolong/PycharmProjects/untitled/iiii'))

'''
分离文件名
分离扩展名

'''
[dirname,filename]=os.path.split('/Users/liuxiaolong/PycharmProjects/untitled/sw724.vaps')
print(dirname,"\n",filename)

[fname,fename]=os.path.splitext('/Users/liuxiaolong/PycharmProjects/untitled/sw724.vaps')
print(fname,"\n",fename)

'''
获得文件路径
获得文件名
获得当前环境
'''
print("get pathname:",os.path.dirname('/Users/liuxiaolong/PycharmProjects/untitled/sw724.vaps'))
print("get filename:",os.path.basename('/Users/liuxiaolong/PycharmProjects/untitled/sw724.vaps'))
print(os.getenv)