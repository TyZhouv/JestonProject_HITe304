# Coding Your Own Object Detection Program
**capture video frames and process them**  
```python
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))  
```
>>>note: ssd-mobilenet-v2 是否可以直接替代为 conver2ONNX 生成的模型
>>>python如何读取的当前文件（是否生成之后，直接放在文件夹，在tx2build的时候直接设置了环境变量）
