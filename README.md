# 基于深度学习的货架场景检测/分类的AI pipeline

retinanets：tensorflow==2.4.0


## Introduction：

基于AI新零售的大发展趋势下，货架管理问题进行解决，可以看作是提高利润的短期目标。同时选用TensorFlow系列快速上手和易于使用的特性，还关注到很多企业大型项目都使用TF进行部署。


## Task

对真实场景的商品进行检测定位与分类，并将模型构建整个成完整的AI SaaS任务进行交付。


## Action

参考链接：https://github.com/fizyr/keras-retinanet

对于货架商品检测模块，项目难点是图片中的商品排列紧密，正负样本不平衡的特性，因此针对性使用RetinaNet来解决。

- RetinaNet
- backbone：ResNet-50
- 超参数：
  - 回归loss：smooth_l1
  - 分类loss：focal
  - 几何增强：数据增强：旋转，平移，随机裁剪，缩放，翻转
  - 色彩增强：对比度，亮度，hue，饱和度
  - 优化器：Adam，lr=1e-5，0.001
  - 学习率：1e-5，连续2 epoch下降就\*0.1
  - Batch size：1
  - step: 90
  - epoch: 50


对于商品识别模块，使用ResNet50作为Backbone，对数据引入数据增强，对网络引入ImageNet预训练权重进行Finetune。

- ResNet-50
- 输出后引入全局均值池化，再接分类器
- 只训练后面的FC层，冻结CNN权重
- 分类：softmax
- 超参数：
  - Adam
  - CE Loss
  - batch size ：32
  - imput size：224
  - epoch 20
  - steps_per_epoch=100
  - validation_steps=10


训练完成后通过OpenCV串联目标检测与分类模块可视化识别结果,**最后**使用Flask搭建支持在线识别和API调用的 AI SaaS，在本地服务器通过curl请求方式实现轻量级的图像识别的请求。成功实现Web端AI Saas生产级部署。


## curl请求

- 启动服务:python manage.py
- 在另一个窗口执行请求：

`curl -H "Content-Type: application/json" --data @body.json http://localhost:9000/tf2/ai_saas`

HTTP请求的头文件里面会标注出这个请求的内容的类型是json，数据是放到了body.json里面，@body.json表示这是一个本地文件的访问方式，所以body.json里面就传入了一个Image的URL

因为用的是本地文件，所以你去发起请求的时候路径就至关重要，记得请求的时候需要走到有body.json文件的目录下



## Challenge：


1、场景复杂

- 货架包括普通货架、冰柜等，
- 先简化问题聚焦正常的货架商品检测

2、商品细粒度识别

- 商品的形态学与图形学特征其实很接近，差别在颜色，并且真实环境清晰度会更差
- 在通用的分类模型后面再接一个细粒度检测模型，可以拿个小模型(MobileNet)去专门训练一个只识别这三个类别的模型，构成一个级联模型

3、规格识别

- 同种商品不同规格，对图像的特征要求更高

- 思路：不能单单依靠图形学特征或者级联模型解决，需要依赖一些其他的信息。

- 步骤：

- 1、用特征融合的方式，用深度学习+机器学习的特征工程，将两种特征融合起来去做判断

- 2、参照物举例：价签，在商品附近的价签是有一个比例大小的对等关系可以去换算的，即不仅要知道小图的大小，还要知道小图里面的商品在实际的物理空间中的大小，通过辅助信息让特征的维度变得丰富。而不是简单通过一个图形学或者分类器来做判断

- 3、如果使用深度学习可以用取巧的方法，比如说通过学习瓶盖的差异可以分开两类。引入注意力模型。甚至把训练集改成只要瓶盖。



## TODO：

货架层数识别，商品细粒度分类，同类不同规格识别