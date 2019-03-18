# 畸变识别


## 项目概览
```
├─learn_code                # 训练模型的相关代码
│  ├─cascades              # opencv人脸检测的模型
│  ├─conf_hub           
│  ├─examples
│  ├─model_out             # pytorch训练生成的模型
│  ├─temp_code
│  ├─video_hub             # 训练素材集散地
│  ├─cv2_Face_Recognize.py # opencv的人脸识别
│  ├─darknet.py            # yolo的python封装
│  ├─ekf_filter.py         # 卡尔曼滤波算法
│  ├─model.py              # pytorch的模型
│  ├─prepare_data.py       # 处理pytrch的训练素材
│  ├─prepare_yolo.py       # 训练yolo的脚本
│  ├─torch_angle_classify.py    # 角度检测的算法
│  ├─train.py              # 训练pytorch
├─ros                       # 机器人的相关代码
│  ├─my_fly_turtle.py      # 机器人运动
│  ├─my_watch_turtle.py    # 机器人捕获数据
├─yolo_bin                  # yolo的运行环境
│  ├─backup                # yolo训练生成的模型
│  ├─my_data               # yolo训练所需的素材
│  └─Yolo_mark             # yolo打标用的软件
├─app.py                    # Ai算法的服务器
├─ip_camera.py              # 通过RTSP协议读摄像头
├─local_config.py           # 本地环境的配置文件
├─service_test.py           # 服务器测试代码
```
## yolo检测目标表

## 预处理感兴趣区域

## 对目标区域进行classfy

## 对结果进行卡尔曼滤波