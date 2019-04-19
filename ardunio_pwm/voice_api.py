#coding:utf-8

from aip.speech import AipSpeech
import os

APP_ID = '10431313'
API_KEY = '20oczabi54hjlwq2gNLI4uGZ'
SECRET_KEY = 'NZRkupSOFeDn2o2bQeFuE4nnMm6LK9EN'
aipSpeech = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def Generate_dialogue(string):
    result = aipSpeech.synthesis(string, 'zh', 1, {
        'vol': 5, 'per': 5,
    })

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open('faster.mp3', 'wb') as f:
            f.write(result)
        os.system("mpg123 faster.mp3")


def ros_answer( outputtext='好的主人，马上为您开块一点'):
    if (u'你是谁') in outputtext:
        Generate_dialogue('我是精益求精小组创造的家庭陪伴智能机器人')
    elif (u'大') in outputtext:
        Generate_dialogue('我出生才半年')
    elif (u'胖') in outputtext:
        Generate_dialogue('120多斤的女孩子太可怕了')
    elif (u'认识') in outputtext:
        Generate_dialogue('是那个很胖的妹子吗？')
    elif (u'劝') in outputtext:
        Generate_dialogue('别吃了，再吃变成猪了')


def mac_say(word):
    os.system("say "+word)


if __name__ == "__main__":
    while True:
        Generate_dialogue('哈哈哈')