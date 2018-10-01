# [使用ConvNets进行集成识别，定位和检测——OverFeatIntegrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229 "使用卷积网络进行集成识别，定位和检测——OverFeatIntegrated Recognition, Localization and Detection using Convolutional Networks") #
（点击标题链接原文https://arxiv.org/abs/1312.6229）

----------
## Abstract摘要 ##
We present an integrated framework for using Convolutional Networks for classification, localization and detection. 我们提出了一个使用卷积网络进行分类，定位和检测的集成框架。

We also introduce a
novel deep learning approach to localization by learning to predict object boundaries.我们还介绍一个
通过学习预测对象约束来实现定位的新型深度学习方法。 

This integrated framework is the winner
of the localization task of the ImageNet Large Scale Visual Recognition Challenge
2013 (ILSVRC2013) and obtained very competitive results for the detection and
classifications tasks.  这个集成框架是赢家
ImageNet大规模视觉识别挑战的定位任务
2013年（ILSVRC2013）并获得了极具竞争力的检测结果
分类任务。

Finally, we release a feature extractor from our best model
called OverFeat.最后，我们从最好的模型中发布了一个特征提取器
叫做OverFeat。
## 1、Introduction简介 ##
Recognizing the category of the dominant object in an image is a tasks to which Convolutional
Networks (ConvNets) [17] have been applied for many years识别图像中主要对象的类别是Convolutional的任务
网络（ConvNets）[17]已被应用多年。

The main advantage of ConvNets for many such tasks is that the entire system is trained end to
end, from raw pixels to ultimate categories, thereby alleviating the requirement to manually design
a suitable feature extractor. 卷积网络的优点：端到端The main disadvantage is their ravenous appetite for labeled training
samples.卷积网络的缺点：依赖于有标签的训练数据集。

The main point of this paper is to show that training a convolutional network to simultaneously
classify, locate and detect objects in images can boost the classification accuracy and the detection
and localization accuracy of all tasks.本文的重点是展示同时训练卷积网络
分类，定位和检测图像中的对象可以提高分类准确度和检测
和所有任务的定位准确性。 The paper proposes a new integrated approach to object
detection, recognition, and localization with a single ConvNet. 本文提出了一种新的对象集成方法
使用单个ConvNet进行检测，识别和定位。We also introduce a novel method for
localization and detection by accumulating predicted bounding boxes. 我们还介绍了一种新颖的方法
通过累积预测的边界框进行定位和检测。

解决图像大小、位置的问题办法：The first idea in addressing this is to apply a ConvNet at multiple
locations in the image, in a sliding window fashion, and over multiple scales.第一个想法是多次应用ConvNet
图像中的位置，滑动窗口方式以及多个比例。This leads to decent
classification but poor localization and detection.  这导致体面
分类但定位和检测不佳。the second idea is to train the system to not
only produce a distribution over categories for each window, but also to produce a prediction of the
location and size of the bounding box containing the object relative to the window.第二个想法是训练系统不
只为每个窗口生成一个类别的分布，而且还产生一个预测
包含相对于窗口的对象的边界框的位置和大小。The third idea is
to accumulate the evidence for each category at each location and size.第三个想法是
在每个位置和尺寸积累对应类别的置信度。

Several authors have also proposed to train ConvNets to directly predict the instantiation parameters
of the objects to be located一些作者还提出训练ConvNets直接预测实例化参数
要定位的对象
Hinton et al. have also proposed
to train networks to compute explicit instantiation parameters of features as part of a recognition
process [12]. Hinton等人。也提出了
训练网络过程中以计算特征的显式实例化参数作为识别的一部分[12]。

Other authors have proposed to perform object localization via ConvNet-based segmentation.其他作者提出通过基于ConvNet的分割来执行对象定位The
simplest approach consists in training the ConvNet to classify the central pixel (or voxel for volumetric images) of its viewing window as a boundary between regions or not [13].该
最简单的方法是训练ConvNet将其观察窗的中心像素（或体积图像的体素）分类为区域之间的边界或不是[13]。semantic segmentation. 语义分割The main idea is to
train the ConvNet to classify the central pixel of the viewing window with the category of the object it belongs to, using the window as context for the decision. 主要想法是
训练ConvNet使用窗口作为决策的上下文，将观察窗口的中心像素分类为它所属的对象的类别。The advantage of this approach is that the bounding contours need not be rectangles, and the regions need
not be well-circumscribed objects. The disadvantage is that it requires dense pixel-level labels for
training. 这种方法的优点是边界轮廓不必是矩形，而区域需要
不是界限清楚的物体。缺点是它需要密集的像素级标签
训练。
## 2、Vision Tasks视觉任务 ##
In this paper, we explore three computer vision tasks in increasing order of difficulty: (i) classification, (ii) localization, and (iii) detection. Each task is a sub-task of the next.在本文中，我们以不断增加的难度顺序探索三种计算机视觉任务：（i）分类，（ii）定位，（iii）检测。 每个任务都是下一个任务的子任务。

classification task 分类任务each image is assigned a single
label corresponding to the main object in the image. 每个图像都分配一个
标签对应于图像中的主要对象。Five guesses are allowed to find the correct
answer (this is because images can also contain multiple unlabeled objects).允许五个猜测找到正确的
回答（这是因为图像还可以包含多个未标记的对象）。 

localization task定位任务a bounding box for the predicted
object must be returned with each guess. 预测的边界框
每次猜测都必须返回对象。

detection task检测任务there can be any number of objects
in each image (including zero)可以存在任意数量的对象
在每个图像（包括零）中

The localization task is a convenient intermediate step between classification and
detection, and allows us to evaluate our localization method independently of challenges specific to
detection (such as learning a background class). 定位任务是分类任务和检测任务的中间步骤
，并允许我们独立于特定的挑战评估我们的定位方法
检测（如学习背景课）。Note that classification and
localization share the same dataset, while detection also has additional data where objects can be
smaller. 注意分类和
定位共享相同的数据集，而检测还具有对象可以的其他数据
小。
![](https://i.imgur.com/5LbGCaA.png)

Figure 1: Localization (top) and detection tasks (bottom). The left images contains our predictions (ordered by decreasing confidence) while the right images show the groundtruth labels. The
detection image (bottom) illustrates the higher difficulty of the detection dataset, which can contain
many small objects while the classification and localization images typically contain a single large
object.图1：定位任务（顶部）和检测任务（底部）。 左图包含我们的预测（通过降低置信度排序），而右图显示地面标签。该
检测图像（底部）说明了可以包含的检测数据集的较高难度
许多小物体，而分类和定位图像通常包含一个大的
目的。
## 3、Classification分类（重点） ##
we improve on the network design and the inference step.我们改进了网络设计和推理步骤。
### 3.1、Model Design and Training模型设计和训练数据 ###
The weights in the network are initialized randomly with  \\(（μ，σ）=（0,1×10^{-2}）\\) .网络中的权重随机初始化为 \\(（μ，σ）=（0,1×10^{-2}）\\) 。updated by stochastic gradient descent通过随机梯度下降更新

Layers 1-5 using rectification (“relu”) non-linearities
and max pooling 1-5层使用整流（“relu”）非线性
和最大池化differences: (i) no contrast normalization is used; (ii)
pooling regions are non-overlapping and (iii) our model has larger 1st and 2nd layer feature maps,
thanks to a smaller stride (2 instead of 4). A larger stride is beneficial for speed but will hurt accuracy.差异：（i）没有使用对比度归一化;（ⅱ）
汇集区域不重叠;（iii）我们的模型具有更大的第1层和第2层特征图，
由于较小的步幅（2取代4）。 较大的步幅有利于速度但会损害准确性。

![](https://i.imgur.com/dVgylED.png)

Table 1: Architecture specifics for fast model. The spatial size of the feature maps depends on
the input image size, which varies during our inference step (see Table 5 in the Appendix). Here
we show training spatial sizes. Layer 5 is the top convolutional layer. Subsequent layers are fully
connected, and applied in sliding window fashion at test time. The fully-connected layers can also
be seen as 1x1 convolutions in a spatial setting. Similar sizes for accurate model can be found in
the Appendix.表1：快速模型的体系结构细节。 要素图的空间大小取决于
输入图像大小，在我们的推理步骤中会有所不同（参见附录中的表5）。 这里
我们展示了训练空间大小。 第5层是顶部卷积层。 后续图层是完全的
连接，并在测试时以滑动窗口方式应用。 完全连接的层也可以
被视为空间环境中的1x1卷积。 可以在中找到类似尺寸的精确模型
附录。

In Fig. 2, we show the filter coefficients from the first two convolutional layers. The first layer filters
capture orientated edges, patterns and blobs. In the second layer, the filters have a variety of forms,
some diffuse, others with strong line structures or oriented edges.在图2中，我们显示了前两个卷积层的滤波器系数。 第一层过滤
捕获定向边缘，图案和斑点。 在第二层，过滤器有多种形式，
一些漫射，其他具有强烈的线结构或定向边缘。

![](https://i.imgur.com/KSgq9mg.png)

Figure 2: Layer 1 (top) and layer 2 filters (bottom).图2：第1层（顶部）和第2层过滤器（底部）。
### 3.2、Feature Extractor提取特征 ###
![](https://i.imgur.com/82a5e3D.png)

Table 2: Classification experiments on validation set. Fine/coarse stride refers to the number of
∆ values used when applying the classifier. Fine: ∆ = 0, 1, 2; coarse: ∆ = 0.表2：验证集的分类实验。 细/粗步幅是指数量
应用分类器时使用的Δ值。 精细：Δ= 0,1,2; 粗：Δ= 0。

![](https://i.imgur.com/FD6YwfY.png)

Figure 4: Test set classification results. During the competition, OverFeat yielded 14.2% top 5
error rate using an average of 7 fast models. In post-competition work, OverFeat ranks fifth with
13.6% error using bigger models (more features and more layers).图4：测试集分类结果。 在比赛期间，OverFeat排名前5位的成绩为14.2％
使用平均7个快速模型的错误率。 在赛后的比赛中，OverFeat排名第五
使用较大型号（更多功能和更多层）的错误率为13.6％。
### 3.3、Multi-Scale Classification多尺度分类 ###
We now explain in detail how the resolution augmentation is performed.我们现在详细解释如何执行分辨率增强。We use 6 scales of input
which result in unpooled layer 5 maps of varying resolution (see Table 5 for details). These are then
pooled and presented to the classifier using the following procedure, illustrated in Fig. 3我们使用6个输入比例
这会产生不同分辨率的未分层第5层图（详见表5）。 然后是这些
汇集并使用以下过程呈现给分类器，如图3所示：
![](https://i.imgur.com/mlovQVW.png)

Figure 3: 1D illustration (to scale) of output map computation for classification, using y-dimension
from scale 2 as an example (see Table 5). (a): 20 pixel unpooled layer 5 feature map. (b): max
pooling over non-overlapping 3 pixel groups, using offsets of ∆ = {0, 1, 2} pixels (red, green, blue
respectively). (c): The resulting 6 pixel pooled maps, for different ∆. (d): 5 pixel classifier (layers
6,7) is applied in sliding window fashion to pooled maps, yielding 2 pixel by C maps for each ∆.
(e): reshaped into 6 pixel by C output maps.图3：使用y维度进行分类的输出地图计算的1D图示（按比例）
以比例2为例（见表5）。 （a）：20像素非池化层5特征图。 （b）：最大
汇集非重叠的3个像素组，使用Δ= {0,1,2}像素的偏移（红色，绿色，蓝色）
分别）。 （c）：得到的6个像素合并的地图，用于不同的Δ。 （d）：5个像素分类器（层
6,7）以滑动窗口的方式应用于合并的地图，为每个Δ产生2个像素乘C的映射。
（e）：重新塑造成6个像素的C输出图。

1. (a) For a single image, at a given scale, we start with the unpooled layer 5 feature maps.（a）对于单个图像，在给定的比例下，我们从未化的第5层特征图开始。
2. (b) Each of unpooled maps undergoes a 3x3 max pooling operation (non-overlapping regions),repeated 3x3 times for (∆x, ∆y) pixel offsets of {0, 1, 2}.（b）每个非池化地图经历3x3最大池化操作（非重叠区域），对于{0,1,2}的（Δx，Δy）像素偏移重复3×3次。
3. (c) This produces a set of pooled feature maps, replicated (3x3) times for different (∆x, ∆y) combinations.（c）这产生一组合并的特征图，对于不同的（Δx，Δy）组合复制（3×3）次。
4. (d) The classifier (layers 6,7,8) has a fixed input size of 5x5 and produces a C-dimensional output vector for each location within the pooled maps. The classifier is applied in sliding-window fashion to the pooled maps, yielding C-dimensional output maps (for a given (∆x, ∆y) combination).（d）分类器（层6,7,8）具有5×5的固定输入大小并产生C维输出合并地图中每个位置的矢量。 分类器应用于滑动窗口时间到合并的地图，产生C维输出图（对于给定的（Δx，Δy）组合）。
5. (e) The output maps for different (∆x, ∆y) combinations are reshaped into a single 3D output map (two spatial dimensions x C classes).（e）将不同（Δx，Δy）组合的输出图重新整形为单个3D输出图（两个空间维度x C类）。

the final classification最后的分类


1. (i) taking the spatial max for each class, at each scale and flip; （i）在每个等级和每个等级中取空间最大值并翻转; 
2. (ii) averaging the resulting C-dimensional vectors from different scales and flips and （ii）平均得到的来自不同尺度和翻转的C维向量和
3. (iii) taking the top-1 or top-5 elements (depending on the evaluation criterion) from the mean class vector.（iii）取得前1或前5元素（取决于评估标准）来自平均类向量。

At an intuitive level, the two halves of the network — i.e. feature extraction layers (1-5) and classifier
layers (6-output) — are used in opposite ways.在直观的层面上，网络的两半 - 即特征提取层（1-5）和分类器
层（6输出） - 以相反的方式使用。
### 3.4、Results结果 ###
### 3.5、ConvNets and Sliding Window Efficiency卷积网络和滑动窗口效率 ###
Note that the last layers of our architecture are fully connected linear layers.请注意，我们架构的最后一层是完全连接的线性层。 At test time, these layers are effectively replaced by convolution operations with kernels of 1x1 spatial extent.在测试时，这些层被1x1空间范围的内核的卷积运算有效地替换。 The entire ConvNet is then simply a sequence of convolutions, max-pooling and thresholding operations exclusively然后，整个ConvNet只是一系列卷积，最大池和阈值操作。
![](https://i.imgur.com/JUPWH7m.png)

Figure 5: The efficiency of ConvNets for detection. During training, a ConvNet produces only a
single spatial output (top). But when applied at test time over a larger image, it produces a spatial
output map, e.g. 2x2 (bottom). Since all layers are applied convolutionally, the extra computation required for the larger image is limited to the yellow regions. This diagram omits the feature
dimension for simplicity.图5：用于检测的ConvNets的效率。 在训练期间，ConvNet仅生成单个空间输出（顶部）。 但是当在测试时在较大的图像上应用时，它产生空间输出图，例如， 2x2（下）。 由于所有层都是卷积应用的，因此较大图像所需的额外计算仅限于黄色区域。 为简单起见，此图省略了特征维度。
## 4、Localization定位（重点） ##
Starting from our classification-trained network, we replace the classifier layers by a regression
network and train it to predict object bounding boxes at each spatial location and scale. 从我们的分类训练网络开始，我们通过回归替换分类器层
网络并训练它以预测每个空间位置和比例的对象边界框。 
### 4.1、Generating Predictions生成预测 ###
To generate object bounding box predictions, we simultaneously run the classifier and regressor
networks across all locations and scales. 为了生成对象边界框预测，我们同时运行分类器和回归量
所有地点和规模的网络。we can assign a confidence to each bounding box.我们可以为每个边界框分配置信度。
### 4.2、Regressor Training回归训练 ###
The regression network takes as input the pooled feature maps from layer 5 回归网络将第5层的池特征映射作为输入。It has 2 fully-connected
hidden layers of size 4096 and 1024 channels, respectively. The final output layer has 4 units which
specify the coordinates for the bounding box edges. As with classification, there are (3x3) copies
throughout, resulting from the ∆x, ∆y shifts. The architecture is shown in Fig. 8.它有2个完全连接
隐藏层大小分别为4096和1024个通道。 最终输出层有4个单位
指定边界框边缘的坐标。 与分类一样，有（3x3）份
整个过程，由Δx，Δy变化产生。 该体系结构如图8所示。
![](https://i.imgur.com/7Q2WXJm.png)

Figure 8: Application of the regression network to layer 5 features, at scale 2, for example. (a)
The input to the regressor at this scale are 6x7 pixels spatially by 256 channels for each of the
(3x3) ∆x, ∆y shifts. (b) Each unit in the 1st layer of the regression net is connected to a 5x5 spatial
neighborhood in the layer 5 maps, as well as all 256 channels. Shifting the 5x5 neighborhood around
results in a map of 2x3 spatial extent, for each of the 4096 channels in the layer, and for each of
the (3x3) ∆x, ∆y shifts. (c) The 2nd regression layer has 1024 units and is fully connected (i.e. the
purple element only connects to the purple element in (b), across all 4096 channels). (d) The output
of the regression network is a 4-vector (specifying the edges of the bounding box) for each location
in the 2x3 map, and for each of the (3x3) ∆x, ∆y shifts.图8：例如，回归网络应用于第2层的第5层特征。 （一个）
此刻度的回归量输入在空间上为6x7像素，每个像素为256个通道
（3×3）Δx，Δy移位。 （b）回归网第一层中的每个单元连接到5x5空间
第5层地图中的邻域，以及所有256个频道。 转移5x5周围的邻居
对于图层中的4096个通道中的每个通道，以及每个通道，得到2x3空间范围的映射
（3×3）Δx，Δy移位。 （c）第二回归层有1024个单位并完全连接（即
紫色元素仅连接到（b）中的紫色元素，跨越所有4096个通道）。 （d）产出
回归网络是每个位置的4向量（指定边界框的边缘）
在2×3图中，对于（3×3）Δx中的每一个，Δy移位。

We fix the feature extraction layers (1-5) from the classification network and train the regression
network using an ℓ2 loss between the predicted and true bounding box for each example. 我们从分类网络中修复特征提取层（1-5）并训练回归
对于每个例子，在预测边界框和真实边界框之间使用ℓ2损失的网络。since the object is mostly outside of these locations, it will be better handled by regression windows
that do contain the object.由于对象大部分位于这些位置之外，因此回归窗口可以更好地处理
确实包含对象。

Training the regressors in a multi-scale manner is important for the across-scale prediction combination.以多尺度方式训练回归量对于跨尺度预测组合非常重要。
### 4.3、Combining Predictions结合预测 ###
We combine the individual predictions (see Fig. 7) via a greedy merge strategy applied to the regressor bounding boxes, using the following algorithm.我们使用以下算法通过应用于回归器边界框的贪婪合并策略组合各个预测（参见图7）。
![](https://i.imgur.com/etfM2kX.png)

![](https://i.imgur.com/Y12ou9l.png)
(a) Assign to\\(C_s\\) the set of classes in the top k for each scale s ∈ 1...6, found by taking the maximum detection class outputs across spatial locations for that scale.

（a）将每个尺度s∈1~6的前k类中的类集分配给\\(C_s\\)，通过在该尺度的空间位置上获取最大检测类输出。

(b) Assign to Bs the set of bounding boxes predicted by the regressor network for each class in \\(C_s\\), across all spatial locations at scale s.

（b）向 Bs 分配由回归网络为\\(C_s\\)中的每个类别预测的边界框集合，跨越所有空间位置，按比例s。

(c) Assign\\(B\leftarrow{\bigcup}sB_s\\)

（c）分配 \\(B\leftarrow{\bigcup}sB_s\\)

(d) Repeat merging until done:

（d）重复合并直到完成：

(e) (b∗ 1, b∗ 2) = argminb16=b2∈Bmatch score(b1, b2)

（e）（b * 1,b * 2）= argminb16 =b2∈B匹配得分（b1，b2）

(f) If match score(b∗ 1, b∗ 2) > t , stop.

（f）如果匹配分数（b * 1，b * 2）> t，则停止。

(g) Otherwise, set B ← B\{b∗ 1, b∗ 2} ∪ box merge(b∗ 1, b∗ 2)

（g）否则，设置B←B \ {b * 1，b * 2}∪框合并（b * 1，b * 2）

In the above, we compute match score using the sum of the distance between centers of the two
bounding boxes and the intersection area of the boxes. box merge compute the average of the
bounding boxes’ coordinates.在上文中，我们使用两者的中心之间的距离之和来计算匹配分数
边界框和框的交叉区域。 框合并计算平均值
边界框的坐标。
![](https://i.imgur.com/XGdR7zX.jpg)

Figure 7: Examples of bounding boxes produced by the regression network, before being combined into final predictions. The examples shown here are at a single scale. Predictions may be
more optimal at other scales depending on the objects. Here, most of the bounding boxes which are
initially organized as a grid, converge to a single location and scale. This indicates that the network
is very confident in the location of the object, as opposed to being spread out randomly. The top left
image shows that it can also correctly identify multiple location if several objects are present. The
various aspect ratios of the predicted bounding boxes shows that the network is able to cope with
various object poses.图7：回归网络生成的边界框示例，然后再将其组合到最终预测中。 这里显示的例子是单一的。 预测可能是
根据物体的不同，在其他尺度上更加优化。 在这里，大多数的边界框都是
最初组织为一个网格，汇聚到一个位置和规模。 这表明网络
对物体的位置非常有信心，而不是随意展开。 左上角
图像显示，如果存在多个对象，它还可以正确识别多个位置。该
预测边界框的各种宽高比表明网络能够应对
各种物体姿势。

合并的边界框的示例，请参见图6
![](https://i.imgur.com/TbSAa1M.jpg)
![](https://i.imgur.com/DDcYQK2.jpg)

Figure 6: Localization/Detection pipeline. The raw classifier/detector outputs a class and a confidence for each location (1st diagram). The resolution of these predictions can be increased using
the method described in section 3.3 (2nd diagram). The regression then predicts the location scale
of the object with respect to each window (3rd diagram). These bounding boxes are then merge and
accumulated to a small number of objects (4th diagram).图6：定位/检测管道。 原始分类器/检测器输出每个位置的类别和置信度（第一个图表）。 可以使用增加这些预测的分辨率
3.3节（第2图）中描述的方法。 回归然后预测位置比例
关于每个窗口的对象（第3图）。 然后合并这些边界框
积累到少量物体（第4张图）。
### 4.4、Experiments实验 ###
![](https://i.imgur.com/laragGG.png)

Figure 9: Localization experiments on ILSVRC12 validation set. We experiment with different
number of scales and with the use of single-class regression (SCR) or per-class regression (PCR).图9：ILSVRC12验证集的定位实验。 我们尝试不同的
使用单级回归（SCR）或每级回归（PCR）进行量表的数量。
![](https://i.imgur.com/Mi75AbY.png)

Figure 10: ILSVRC12 and ILSVRC13 competitions results (test set). Our entry is the winner of
the ILSVRC13 localization competition with 29.9% error (top 5). Note that training and testing data
is the same for both years. The OverFeat entry uses 4 scales and a single-class regression approach.图10：ILSVRC12和ILSVRC13竞赛结果（测试集）。 我们的参赛作品是获胜者
ILSVRC13定位竞争错误率为29.9％（前5名）。 请注意培训和测试数据
这两年都是一样的。 OverFeat条目使用4个比例和单类回归方法。

Our multiscale and multi-view approach was critical to obtaining good performance我们的多尺度和多视图方法对于获得良好性能至关重要
## 5、Detection检测 ##
Detection training in a spatial manner.检测训练以空间方式。Multiple location of
an image may be trained simultaneously.多个位置
可以同时训练图像。Since the model is convolutional, all weights are shared
among all locations. 由于模型是卷积的，因此所有权重都是共享的
在所有地点之间。The main difference with the localization task, is the necessity to predict a
background class when no object is present.与定位任务的主要区别在于预测a的必要性
没有对象时的背景类。

use an initial segmentation step to reduce candidate windows from approximately
200,000 to 2,000. 使用初始分割步骤从大约减少候选窗口的系统
200,000到2,000。

![](https://i.imgur.com/MI5FZc2.png)

Figure 11: ILSVRC13 test set Detection results. During the competition, UvA ranked first with
22.6% mAP. In post competition work, we establish a new state of the art with 24.3% mAP. Systems
marked with * were pre-trained with the ILSVRC12 classification data.图11：ILSVRC13测试集检测结果。 在比赛期间，UvA排名第一
mAP为22.6％。 在竞赛后的工作中，我们以24.3％的mAP建立了一种新的技术水平。系统
用*用ILSVRC12分类数据进行预训练。
## 6、Discussion总结 ##
We have presented a multi-scale, sliding window approach that can be used for classification, localization and detection.我们提出了一种多尺度滑动窗口方法，可用于分类，定位和检测。A second important contribution of our paper
is explaining how ConvNets can be effectively used for detection and localization tasks.本文的第二个重要贡献
正在解释如何将ConvNets有效地用于检测和定位任务。We have proposed an integrated pipeline that can
perform different tasks while sharing a common feature extraction base, entirely learned directly
from the pixels.我们已经提出了一个可以的综合管道
在共享一个共同的特征提取基础时执行不同的任务，完全直接学习
从像素。

Our approach might still be improved in several ways. (i) For localization, we are not currently
back-propping through the whole network; doing so is likely to improve performance. (ii) We are
using ℓ2 loss, rather than directly optimizing the intersection-over-union (IOU) criterion on which
performance is measured. Swapping the loss to this should be possible since IOU is still differentiable, provided there is some overlap. (iii) Alternate parameterizations of the bounding box may
help to decorrelate the outputs, which will aid network training.我们的方法可能会在几个方面得到改善。 （i）对于定位，我们目前不是
通过整个网络反击; 这样做可能会提高性能。 （ii）我们是
使用ℓ2损失，而不是直接优化其上的交叉联合（IOU）标准
绩效是衡量的。 如果存在一些重叠，因此IOU仍然是不同的，因此应该可以将损失交换到此。 （iii）边界框的替代参数化可以
有助于对输出进行去相关，这将有助于网络培训。
## Appendix: Additional Model Details附录：附加模型详细信息 ##
![](https://i.imgur.com/0Agjoi1.png)

Table 3: Architecture specifics for accurate model. It differs from the fast model mainly in the
stride of the first convolution, the number of stages and the number of feature maps.表3：精确模型的体系结构细节。 它与快速模型的不同之处主要在于
第一卷积的步幅，阶段的数量和特征映射的数量。

![](https://i.imgur.com/R4WaQRO.png)

Table 4: Number of parameters and connections for different models.表4：不同型号的参数和连接数。

![](https://i.imgur.com/raWHd7m.png)

Table 5: Spatial dimensions of our multi-scale approach. 6 different sizes of input images are
used, resulting in layer 5 unpooled feature maps of differing spatial resolution (although not indicated in the table, all have 256 feature channels). The (3x3) results from our dense pooling operation
with (∆x, ∆y) = {0, 1, 2}. See text and Fig. 3 for details for how these are converted into output
maps.表5：我们的多尺度方法的空间维度。 6种不同尺寸的输入图像
使用时，得到不同空间分辨率的第5层非池化特征图（尽管表中没有说明，但都有256个特征通道）。 （3x3）来自我们的密集池操作
（Δx，Δy）= {0,1,2}。 有关如何将这些转换为输出的详细信息，请参见文本和图3
地图。

感谢：

1. [深度学习研究理解6:OverFeat:Integrated Recognition, Localization and Detection using Convolutional Networks](https://blog.csdn.net/whiteinblue/article/details/43374195 "深度学习研究理解6:OverFeat:Integrated Recognition, Localization and Detection using Convolutional Networks")

2. [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks(阅读)](https://blog.csdn.net/langb2014/article/details/52334490 "OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks(阅读)")

3. [OverFeat Integrated Recognition, Localization and Detection using Convolutional Networks](https://yufeigan.github.io/2014/11/23/OverFeat-Integrated-Recognition-Localization-and-Detection-using-Convolutional-Networks/ "OverFeat Integrated Recognition, Localization and Detection using Convolutional Networks")

4. [[转载]深度学习论文笔记：OverFeat](http://azraelzhu.w159.mc-test.com/index.php/overfeat/ "[转载]深度学习论文笔记：OverFeat")

5. [【一步一步的积累】OverFeat](https://blog.csdn.net/seavan811/article/details/49825891 "【一步一步的积累】OverFeat")