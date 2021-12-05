# IMVFX_Final
The final project of Image Manipulation Techniques and Visual Effects course in NYCU

## 如何設計GUI [Ref](https://www.itread01.com/content/1547572153.html)
1. 使用Qt Designer設計互動介面後儲存成 .ui 檔 ( anaconda 有內建，使用方法[參考](http://elmer-storage.blogspot.com/2018/04/pyqt.html))
2. 執行一下 command 產生 GUI 的 python 檔 , 
> formate : python -m PyQt5.uic.pyuic -o [fileName.py] [fileName.ui]
```
python -m PyQt5.uic.pyuic -o UI/MainGUI.py UI/MainGUI.ui
```

## 教學
### [pyqt document](https://doc.qt.io/)
### pyqt + opencv [ref](https://shengyu7697.github.io/python-opencv-show-image-pyqt/)
### override function at instance level [ref](https://stackoverflow.com/questions/394770/override-a-method-at-instance-level)
### OpenCV document [ref](https://docs.opencv.org/4.x/index.html)

## 也許可用的 Model
### object detection
1. Detectron2 [ref](https://yanwei-liu.medium.com/mask-r-cnn-with-detectron2-20c8f67b7f48)
2. yolo

### object tracking
1. yolo + deep sort [ref](https://peaceful0907.medium.com/%E5%88%9D%E6%8E%A2%E7%89%A9%E4%BB%B6%E8%BF%BD%E8%B9%A4-multiple-object-tracking-mot-4f1b42e959f9)