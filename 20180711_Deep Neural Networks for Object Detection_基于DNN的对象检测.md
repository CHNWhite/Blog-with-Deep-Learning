# [Deep Neural Networks for Object Detection用于物体检测的深度深度神经网络](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf "Deep Neural Networks for Object Detection用于物体检测的深度深度神经网络") #
（点击标题链接原文http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf）

----------
## 1、Introduction介绍 ##
### DNNs与传统分类方法区别 ###
1. First, they are deep
architectures which have the capacity to learn more complex models than shallow ones [2].深度模型架构。
2. This
expressivity and robust training algorithms allow for learning powerful object representations without the need to hand design features.不需要手工设计特征。

In this paper, we exploit the power of DNNs for the problem of object detection, where we not only
classify but also try to precisely localize objects.这篇论文解决了物体检测的分类问题和精确定位对象。
We present a formulation which is capable of predicting the bounding boxes of multiple objects in
a given image. 能够预测多个物体的边界框。
![](https://i.imgur.com/yxogB5O.png)
Figure 1: A schematic view of object detection as DNN-based regression.图1:作为基于DNN的回归的对象检测的原理图。
![](https://i.imgur.com/id6x2Fa.png)
Figure 2: After regressing to object masks across several scales and large image boxes, we perform
object box extraction. The obtained boxes are refined by repeating the same procedure on the sub
images, cropped via the current object boxes. For brevity, we display only the full object mask,
however, we use all five object masks.图2：在几个刻度和大图像框上回归到对象蒙版后，我们执行
对象框提取。 通过在子上重复相同的过程来细化所获得的框
图像，通过当前对象框裁剪。 为简洁起见，我们只显示完整的对象蒙版，
但是，我们使用所有五个对象蒙版。

In this paper, we demonstrate that DNN-based regression is capable of learning features which
are not only good for classification, but also capture strong geometric information. 证明：分类，捕获几何信息。

## 2、Related Work相关工作 ##
物体检测研究最广泛的范例之一the deformable part-based model可变性部件的模型，pictorial structure.图案结构It can be considered as a 2-layer model – parts being
the first layer and the star model being the second layer.2层模型，部分是第一层，星形是第二层。DNNs图层是通用的。

Deep architectures for object detection and parsing have been motivated by part-based models and
traditionally are called compositional models，用于对象检测和解析的深层体系结构受到基于部件的模型的推动传统上称为组合模型，where the object is expressed as layered composition
of image primitives.其中对象表示为分层组合
图像基元。

 
## 3、DNN-based Detection基于DNN的探测 ##
The core of our approach is a DNN-based regression towards an object mask核心是基于DNN的对象掩码回归。（图1）
## 4、Detection as DNN Regression  DNN回归检测 ##
Our network is based on the convolutional DNN defined by [14]. It consists of total 7 layers, the
first 5 of which being convolutional and the last 2 fully connected. Each layer uses a rectified linear
unit as a non-linear transformation. Three of the convolutional layers have in addition max pooling.
For further details, we refer the reader to [14]我们的网络基于[14]定义的卷积DNN。 它由总共7层组成其中前5个是卷积的，后2个是完全连接的。 每层使用整流线性单位作为非线性变换。 其中三个卷积层还有最大池。有关详细信息，请参阅[14]

![](https://i.imgur.com/6UBn1rG.png)

 Θ are the parameters of the network and N is the total number of pixels. Θ是网络的参数，N是像素的总数。
![](https://i.imgur.com/flaUu1b.png)

The network is trained by minimizing the L2 error for predicting a ground truth mask m 2 [0; 1]N
for an image x:通过最小化L2误差来训练网络以预测地面实况掩模m 2 [0;1] n的对于图像x：

where the sum ranges over a training set D of images containing bounding boxed objects which are
represented as binary masks其中总和范围超过包含边界框对象的图像的训练集D.表示为二进制掩码
## 5、Precise Object Localization via DNN-generated Masks基于DNN生成掩模的精确目标定位 ##
论文这部分只要对三个具有挑战性的问题进行分析和解决：

First, a single object mask might not be sufficient to disambiguate objects
which are placed next to each other.第一，单个对象掩码可能不足以消除彼此相邻的对象的歧义。

 Second, due to the limits in the output size, we generate masks
that are much smaller than the size of the original image.第二，由于输出大小的限制，生成掩码比原始图像尺寸小得多。（would be insufficient to
precisely localize an object不足以精确定位一个对象）。

Finally, since we use as an input the full
image, small objects will affect very few input neurons and thus will be hard to recognize.第三，输入整张图片，小物体对输入神经元的影响很少，导致很难识别。

In the
following, we explain how we address these issues.下面将解释如何解决这些问题。
### 5.1、Multiple Masks for Robust Localization 用于稳健本地化的多个掩码 ###
$$m^h,h\in\lbrace full,bottom,top,left,right \rbrace $$
Further, if two objects of the same type are placed next to each other, then at least two of the
produced five masks would not have the objects merged which would allow to disambiguate them.此外，如果相同类型的两个对象彼此相邻放置，则至少两个
生成五个掩码不会合并对象，这将允许消除它们的歧义。

Denote by T (i; j) the rectangle in the image for which the presence of an object is
predicted by output (i; j) of the network. This rectangle has upper left corner at (dd1 (i−1); dd2 (j−1))
and has size d1
d × dd1 , where d is the size of the output mask and d1; d2 the height and width of the
image. During training we assign as value m(i; j) to be predicted as portion of T (i; j) being covered
by box bb(h) :定义T(i,j)为模型输出值(i,j)对应原始图片中小方格的大小,则这个小方框左上角的坐标为\\(\frac{d_1}{d}(i-1)\\)

，\\(\frac{d_2}{d}(j-1)\\)

大小为

\\(\frac{d_1}{d}\\)
\\(\times\\)

\\(\frac{d_2}{d}\\)

d是输出掩码的大小，$d_1$,$d-2$是输入图像的高和宽。在训练时利用模型输出的m(i,j)预测原始图像对应的部分T(i,j)被box bb(h)覆盖
$$ m^h(i,j;bb)=\frac{area(bb(h)\cap T(i,j))}{area(T(i,j))} \ \ \ \ \ \ \ \ \ \ \ \  (1)$$
其中 bb(full) 对应Ground Truth Object Box，其余的 bb(h) 对应余下的4个Original Box Halves。 
注意：作者使用Full Box 以及 Top, Bottom, Left, Right Halves of the Box代表5种不同的覆盖类型。计算的结果 mh(bb) 对应于Ground Truth Box bb，它在训练时被用来代表模型输出的类型 h。
one could train one network for all masks where the output
layer would generate all five of them.可以为所有Masks训练一个网络，其输出层将生成它们中的所有5个masks
 In this way, the five localizers
would share most of the layers and thus would share features,5个定位器将共享大多数层，因此能够共享特征。using the same localizer for a
lot of distinct classes 使用相同的定位器为很多不同的类进行定位
### 5.2、Object Localization from DNN Output DNN输出的对象本地化 ###
计算分数S的公式如下：

$$ S(bb,m)=\frac{1}{arcea(bb)}\sum_{(i,j)}m(i,j)area(bb\cap T(i,j))\ \ \ \ \ \ \ \ \ \ \ \  (2) $$

最后得到的分数S（bb）如下：

$$ S(bb)=\sum_{h\in halves}(S(bb(h),m^h)-S(bb(\overline{h}),m^h))\ \ \ \ \ \ \ \ \ \ \ \  (3)$$

halves = ffull; bottom; top; left; leftg index the full box and its four halves. For h denoting
one of the halves h¯ denotes the opposite half of h 其中 \\( m^h,h\in\lbrace full,bottom,top,left,right \rbrace \\) ，h是halves中的一个，h被定义为与h相反的一半。 
### 5.3、Multi-scale Refinement of DNN Localizer DNN定位器的多尺度细化 ###
The issue with insufficient resolution of the network output is addressed in two ways: (i) applying
the DNN localizer over several scales and a few large sub-windows; (ii) refinement of detections by
applying the DNN localizer on the top inferred bounding boxes 网络输出的Binary Masks分辨率不足的问题以两种方式解决：


1. （i）将DNN Localizer应用于若干Scales和几个大Sub-Windows; 


1. （ii）通过在顶部推断的Bounding Boxes上应用DNN Localizer来改进检测（参见 Fig. 2）。 

To achieve the above goals, we use three scales: the full image and two other scales such that the
size of the window at a given scale is half of the size of the window at the previous scale. We cover
the image at each scale with windows such that these windows have a small overlap – 20% of their
area. These windows are relatively small in number and cover the image at several scales. Most
importantly, the windows at the smallest scale allow localization at a higher resolution.为了实现上述目标，我们使用三个Scales：完整图像和两个其他Scales，使得在给定Scale下的窗口的尺寸是先前给定Scale窗口尺寸的一半。 我们用每个Scale的窗口覆盖图像，使得这些窗口具有小的重叠 - 其面积的20％。 这些窗口在数量上相对较小并且在几个尺度上覆盖图像。 最重要的是，最小尺度的窗口允许以更高的分辨率定位。 

![](https://i.imgur.com/ys6DSmM.png)
## 6、DNN TrainingDNN训练 ##
 it needs
to be trained with a huge amount of training data: objects of different sizes need to occur at almost
every location它需要接受大量训练数据训练：不同大小的物体几乎需要发生每个地方。

60% negative and 40% positive samples60%负样本40%正样本The negative samples are those whose bounding boxes have less than 0.2 Jaccard-similarity
with any of the groundtruth object boxes The positive samples must have at least 0.6 similarity with
some of the object bounding boxes and are labeled by the class of the object with most similar
bounding box to the crop. 负样本是其边界框具有小于0.2的Jaccard相似性的样本
与任何groundtruth对象框正面样本必须具有至少0.6相似性
一些对象边界框并由最相似的对象类标记
作物的边界框。


## 7、Experiments实验设计 ##
Dataset数据集：  [VOC2007/2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/ "VOC2007/2012")   We use precision-recall curves and
average precision (AP) per class to measure the performance of the algorithm.我们使用精确回忆曲线和
每个类的平均精度（AP）来衡量算法的性能。

Evaluation评估： The first is a sliding window version
of a DNN classifier第一个是滑动窗口版本的DNN分类器。（VOC2007）

The second approach is the 3-layer compositional model第二种方法是3层组合模型（VOC2011）

Contrary to the widely cited DPM 与广泛引用的DPM方法相反 DetectorNet excels at deformable objects such
as bird, cat, sheep, dog.  DetectorNet擅长可变形物体

![](https://i.imgur.com/O500nMZ.jpg)
Figure 3: For each image, we show two heat maps on the right: the first one corresponds to the
output of DNNfull, while the second one encodes the four partial masks in terms of the strength of
the colors red, green, blue and yellow. In addition, we visualize the estimated object bounding box.
All examples are correct detections with exception of the examples in the last row.
图3：对于每个图像，我们在右边显示两个热图：第一个对应于DNNULL的输出，而第二个在强度方面编码四个部分掩模。红色、绿色、蓝色和黄色。此外，我们可视化估计对象边界框。除了最后一行中的示例之外，所有示例都是正确的检测。

Finally, the refinement step contributes drastically to the quality of the detection. This can be seen in
Fig. 4 where we show the precision vs recall of DetectorNet after the first stage of detection and after
refinement. A noticeable improvement can be observed, mainly due to the fact that better localized
true positives have their score boosted.最后，细化步骤极大地有助于检测质量。 这可以在
图4显示了第一阶段检测后及之后DetectorNet的精确度与召回率
细化。 可以观察到明显的改善，主要是由于更好的局部化
真正的积极因素提升了他们的分数。
![](https://i.imgur.com/F5sWC0O.png)
Figure 4: Precision recall curves of DetectorNet after the first stage and after the refinement.图4：第一阶段后和精化后的DetectorNet的精确回忆曲线。
## 8、Conclusion结论 ##
We show that the simple
formulation of detection as DNN-base object mask regression can yield strong results when applied
using a multi-scale course-to-fine procedure. 我们表明这很简单作为DNN基对象掩模回归的检测公式可以在应用时产生强烈的结果使用多尺度的课程到精细程序。

附件：[1、前辈在CSDN的解读](https://blog.csdn.net/qingqingdeaini/article/details/53099468?locationNum=16&fps=1 "1、前辈在CSDN的解读")