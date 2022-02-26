# Collecting my own Detection Datasets
**[待插入图片 自己使用camera Tool]**
![camera-capture tool](https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/pytorch-collection-detect.jpg)

## Pascal Voc Dataset Structure
+ JPEGImages：存放的是训练与测试的所有图片。
+ Annotations：里面存放的是每张图片打完标签所对应的XML文件
+ ImageSets：ImageSets文件夹下本次讨论的只有Main文件夹，此文件夹中存放的主要又有四个文本文件test.txt,train.txt,trainval.txt,val.txt,其中分别存放的是测试集图片的文件名、训练集图片的文件名、训练验证集图片的文件名、验证集图片的文件名。
+ SegmentationClass与SegmentationObject：存放的都是图片，且都是图像分割结果图，对目标检测任务来说没有用。class segmentation 标注出每一个像素的类别 。object segmentation 标注出每一个像素属于哪一个物体  

**[待插入图片 我的数据集目录 对应解释 代码块更改为我的xml文件]**
```python
<annotation>
  <folder>17</folder> # 图片所处文件夹
  <filename>77258.bmp</filename> # 图片名
  <path>~/frcnn-image/61/ADAS/image/frcnn-image/17/77258.bmp</path>
  <source>  #图片来源相关信息
    <database>Unknown</database>  
  </source>
  <size> #图片尺寸
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>  #是否有分割label
  <object> 包含的物体
    <name>car</name>  #物体类别
    <pose>Unspecified</pose>  #物体的姿态
    <truncated>0</truncated>  #物体是否被部分遮挡（>15%）
    <difficult>0</difficult>  #是否为难以辨识的物体， 主要指要结体背景才能判断出类别的物体。虽有标注， 但一般忽略这类物体
    <bndbox>  #物体的bound box
      <xmin>2</xmin>
      <ymin>156</ymin>
      <xmax>111</xmax>
      <ymax>259</ymax>
    </bndbox>
  </object>
</annotation>
```
![实际采集数据集代码截图]()

# Training Model
```python
cd jetson-inference/python/training/detection/ssd
python3 train_ssd.py --dataset-type=voc --data=data/<YOUR-DATASET> --model-dir=models/<YOUR-MODEL>
```
# Convert PyTorch model to ONNX  
python3 onnx_export.py --model-dir=models/<YOUR-MODEL>  
> note:转换后的模型将保存在 <YOUR-MODEL>/ssd-mobilenet.onnx  

# Load Model
```python
NET=models/<YOUR-MODEL>
detectnet --model=$NET/ssd-mobilenet.onnx --labels=$NET/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /vedio/video0
```
![Demo Detect结果截图]()
  
