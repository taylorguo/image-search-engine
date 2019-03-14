# Image Search Engine

使用OpenCV计算RGB颜色直方图

获得颜色特征描述子后，卡方距离计算相似度

## Save descriptors to disk

base_search.py 将图像库的描述子存入.cpickle文件中

## Compare unidex picture to descriptors

将新图片与图像库中的描述子对比距离

这里用来计算MTCNN识别出来的人脸与目标人脸是否类似

如果距离阈值大于float(1)，认为是不同人脸

对人脸进行分类，获取目标人脸后，进行变脸(deepfake / GAN)

=======================================

# [Perceptual Image Hash](http://phash.org/docs/pubs/thesis_zauner.pdf)
