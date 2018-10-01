# [用于精确对象检测和语义分割的丰富特征层次—Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524 "用于精确对象检测和语义分割的丰富特征层次—Rich feature hierarchies for accurate object detection and semantic segmentation") #
（点击标题链接原文https://arxiv.org/abs/1311.2524）
## ↓ 辅助阅读，帮助理解 ↓ ##
[r-cnn-ilsvrc2013-workshop.pdf](http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf "r-cnn-ilsvrc2013-workshop.pdf")
----------
## Abstract ##
The best-performing methods are complex ensemble systems that typically combine multiple low-level
image features with high-level context.物体检测：表现最佳的方法是复杂的集合系统，通常将多个低级图像特征与高级上下文相结合。In this paper, we
propose a simple and scalable detection algorithm在本文中，我们提出了一种简单且可扩展党的检测算法。 Our approach combines two key insights:
(1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to
localize and segment objects and (2) when labeled training
data is scarce, supervised pre-training for an auxiliary task,
followed by domain-specific fine-tuning, yields a significant
performance boost.我们的方法结合了两个关键的见解：（1）高容量卷积神经网络（CNN）可以应用于自下而上的区域提议，以定位和分割对象; （2）当标记的训练数据很少时当监督辅助任务的预训练然后执行特定领域的微调时，可以显着提高性能。Since we combine region proposals
with CNNs, we call our method R-CNN: Regions with CNN
features. 由于我们将候选区域与CNN结合起来，我们将该方法称为R-CNN：具有CNN功能的区域。
## 1. Introduction ##
Features matter. 特征是重要的。recognition occurs several
stages downstream, which suggests that there might be hierarchical, multi-stage processes for computing features that
are even more informative for visual recognition.识别发生在下游几个阶段，这表明可能存在用于计算特征的分层的多阶段过程，多极化法计算的特点，甚至是更丰富的视觉识别。

The neocognitron, however, lacked a supervised training
algorithm. 然而，新认知缺乏监督训练算法。stochastic gradient descent via backpropagation was effective for training convolutional neural
networks (CNNs), a class of models that extend the neocognitron.通过反向传播的随机梯度下降对于训练卷积神经网络（CNNs）是一种有效的训练方法，CNN是一类扩展新神经元的模型。

The central
issue can be distilled to the following: To what extent do
the CNN classification results on ImageNet generalize to
object detection results on the PASCAL VOC Challenge? 中心问题可以归结为以下几点：ImageNet的CNN分类结果在多大程度上推广到PASCAL VOC挑战的对象检测结果？

We answer this question by bridging the gap between
image classification and object detection.我们解决了这个问题，弥补了图像分类和目标检测之间的差距。  we focused on two problems: localizing objects
with a deep network and training a high-capacity model
with only a small quantity of annotated detection data.我们专注于两个问题：使用深度网络定位对象和训练一个高容量的模型，只有少量的注释检测数据。

Unlike image classification, detection requires localizing (likely many) objects within an image.与图像分类不同，检测需要在图像内定位（可能多个）对象。One approach
frames localization as a regression problem. 一种方法将定位框架化为回归问题。An alternative is to build a
sliding-window detector.另一种方法是建立一个滑动窗口检测器。In order
to maintain high spatial resolution, these CNNs typically
only have two convolutional and pooling layers.为了保持高空间分辨率，这些CNNs通常只具有两个卷积和池化层。However,
units high up in our network, which has five convolutional
layers, have very large receptive fields (195 × 195 pixels)
and strides (32×32 pixels) in the input image, which makes
precise localization within the sliding-window paradigm an
open technical challenge.然而，在我们的网络中具有五个卷积层的单位在输入图像中具有非常大的接受域（195×195像素）和步幅（32×32像素），这使得在滑动窗口范式内的精确定位成为一个开放的技术挑战。

Instead, we solve the CNN localization problem by operating within the “recognition using regions” paradigm [21],
which has been successful for both object detection [39] and
semantic segmentation [5]. 取而代之的是，我们通过在“识别使用区域”范式中操作来解决CNN定位问题，该范式已成功用于对象检测和语义分割。 extracts a fixed-length feature vector from
each proposal using a CNN使用CNN从每个候选中提取固定长度的特征向量 classifies each region
with category-specific linear SVMs使用类别特定的线性SVM对每个区域进行分类。Since our system combines
region proposals with CNNs, we dub the method R-CNN:
Regions with CNN features.由于我们的系统将候选区域与CNNs结合起来，我们称之为R-CNN方法：具有CNN特征的区域。

![](https://i.imgur.com/S6WjLTo.png)

Figure 1: Object detection system overview. Our system (1) takes an input image, (2) extracts around 2000 bottom-up region proposals, (3) computes features for each proposal using a large convolutional neural network (CNN), and then (4) classifies each region using class-specific linear SVMs. R-CNN achieves a mean average precision (mAP) of 53.7% on PASCAL VOC 2010. For comparison, [39] reports 35.1% mAP using the same region proposals, but with a spatial pyramid and bag-of-visual-words approach. The popular deformable part models perform at 33.4%. On the 200-class ILSVRC2013 detection dataset, R-CNN’s mAP is 31.4%, a large improvement over OverFeat [34], which had the previous best result at 24.3%.图1：对象检测系统概述。 我们的系统（1）采用输入图像，（2）提取大约2000个自下而上区域提议，（3）使用大型卷积神经网络（CNN）计算每个提议的特征，然后（4）使用类对每个区域进行分类 特定的线性SVM。 R-CNN在PASCAL VOC 2010上实现了53.7％的平均精确度（mAP）。相比之下，[39]使用相同的区域位置报告了35.1％的mAP，但是具有空间金字塔和bag-of-visual-words方法。 流行的可变形部分模型的性能为33.4％。 在200级ILSVRC2013检测数据集中，R-CNN的mAP为31.4％，比OverFeat [34]有很大改进，其中之前的最佳结果为24.3％。

A second challenge faced in detection is that labeled data is scarce and the amount currently available is insufficient for training a large CNN. 检测面临的第二个挑战是标记数据稀缺，目前可用的数量不足以培训大型CNN。The conventional solution to this
problem is to use unsupervised pre-training, followed by supervised fine-tuning.该问题的传统解决方案是使用无监督的预训练，然后进行有监督的微调。The second principle contribution of this paper is to show that supervised pre-training
on a large auxiliary dataset (ILSVRC), followed by domainspecific fine-tuning on a small dataset (PASCAL), is an
effective paradigm for learning high-capacity CNNs when
data is scarce. 本文的第二个原则是，显示对大型辅助数据集（ILSVRC）进行有监督的预训练，然后对小型数据集（PASCAL）进行域特定微调，是学习高级数据集的有效范例。数据稀缺时的容量CNN。Krizhevsky’s CNN can be used (without finetuning) as a blackbox feature extractor, yielding excellent
performance on several recognition tasks表明Krizhevsky的CNN可以作为黑盒特征提取器使用（没有精细调整），在几个识别任务上产生出色的表现。

The only class-specific
computations are a reasonably small matrix-vector product
and greedy non-maximum suppression. 唯一的类特定计算是相当小的矩阵向量乘积和贪婪的非最大抑制。

we demonstrate that
a simple bounding-box regression method significantly reduces mislocalizations, which are the dominant error mode.我们证明了一种简单的边界框回归方法可以显着减少错误定位，这是主要的错误模式。
## 2. Object detection with R-CNN 使用R-CNN进行物体检测 ##
Our object detection system consists of three modules.我们的物体检测系统由三个模块组成。
The first generates category-independent region proposals.
These proposals define the set of candidate detections available to our detector. 第一个生成与类别无关的候选框，这些候选定义了我们的探测器可用于候选检测集。The second module is a large convolutional neural network that extracts a fixed-length feature
vector from each region. 第二个模块是一个大型卷积神经网络，它从每个区域提取固定长度的特征向量。The third module is a set of classspecific linear SVMs. 第三个模块是一组特定于类别的线性SVM。
### 2.1. Module design 模型设计 ###
Region proposals候选框：methods for generating category-independent region proposals.生成与类别无关的候选框的方法。Examples include: objectness [1], selective search [39],
category-independent object proposals [14], constrained
parametric min-cuts (CPMC) [5], multi-scale combinatorial
grouping [3], and Cires¸an et al例子包括：对象性[1]，选择性搜索[39]，类别无关的对象提议[14]，约束参数最小割（CPMC）[5]， 多尺度组合分组[3]和Cires¸an等。

Feature extraction特征提取Features are computed by forward propagating
a mean-subtracted 227 × 227 RGB image through five convolutional layers and two fully connected layers.通过向前传播平均减去的227×227 RGB图像通过五个卷积层和两个完全连接的层来计算特征。
![](https://i.imgur.com/vqnvoId.png)

Figure 2: Warped training samples from VOC 2007 train.图2：来自VOC 2007的扭曲训练样本。（扭曲方案见附录A）
### 2.2. Test-time detection 测试时间检测 ###
Run-time analysis运行时间分析Two properties make detection efficient. 两个属性是的检测有效。First, all CNN parameters are shared across all categories. 第一，所有类别共享所有CNN参数。Second, the feature vectors computed by the CNN are low-dimensional when compared to other common approaches, such as spatial pyramids with bag-of-visual-word
encodings.第二，与其他常见方法（例如具有视觉字编码的空间金字塔）相比，由CNN计算的特征向量是低维的。

The result of such sharing is that the time spent computing region proposals and features (13s/image on a GPU
or 53s/image on a CPU) is amortized over all classes. 这种共享的结果是，计算候选框和特征（在GPU上每张图13s，在CPU上每张图53s）所花费的时间在所有类上摊销。

R-CNN can scale to thousands
of object classes without resorting to approximate techniques R-CNN可以扩展到数千个对象类，而无需采用近似法。
### 2.3. Training ###
Supervised pre-training监督训练

Domain-specific fine-tuning特定领域的微调

Object category classifiers对象类别分类器We resolve this issue with an IoU overlap threshold我们用IoU重叠阈值来解决这个问题。

In Appendix B we discuss why the positive and negative examples are defined differently in fine-tuning versus SVM training. We also discuss the trade-offs involved in training detection SVMs rather than simply using the outputs from
the final softmax layer of the fine-tuned CNN.在附录B中，我们讨论了为什么在微调和SVM训练中对正面和负面示例的定义不同。 我们还讨论了训练检测SVM所涉及的权衡，而不是简单地使用来自的输出微调CNN的最终softmax层。
### 2.4. Results on PASCAL VOC 2010-12 ###
![](https://i.imgur.com/3Dbornz.png)

Table 1: Detection average precision (%) on VOC 2010 test. R-CNN is most directly comparable to UVA and Regionlets since all
methods use selective search region proposals. Bounding-box regression (BB) is described in Section C. At publication time, SegDPM
was the top-performer on the PASCAL VOC leaderboard. yDPM and SegDPM use context rescoring not used by the other methods.表1：VOC 2010测试的检测平均精度（％）。 R-CNN最直接可与UVA和Regionlet相媲美
方法使用选择性搜索区域提议。 边界框回归（BB）在C部分中描述。在发布时，SegDPM
是PASCAL VOC排行榜的最佳表现者。 yDPM和SegDPM使用其他方法未使用的上下文重新绑定。
### 2.5. Results on ILSVRC2013 detection ###
![](https://i.imgur.com/fZS6DGU.png)

Figure 3: (Left) Mean average precision on the ILSVRC2013 detection test set. Methods preceeded by * use outside training data
(images and labels from the ILSVRC classification dataset in all cases). (Right) Box plots for the 200 average precision values per
method. A box plot for the post-competition OverFeat result is not shown because per-class APs are not yet available (per-class APs for
R-CNN are in Table 8 and also included in the tech report source uploaded to arXiv.org; see R-CNN-ILSVRC2013-APs.txt). The red
line marks the median AP, the box bottom and top are the 25th and 75th percentiles. The whiskers extend to the min and max AP of each
method. Each AP is plotted as a green dot over the whiskers (best viewed digitally with zoom).图3：（左）ILSVRC2013检测测试装置的平均平均精度。 方法之前*使用外部训练数据
（在所有情况下来自ILSVRC分类数据集的图像和标签）。 （右）每个200平均精度值的箱形图
方法。 没有显示赛后OverFeat结果的方框图，因为每类AP尚不可用（每类AP为
R-CNN见表8，也包含在上传到arXiv.org的技术报告来源中; 见R-CNN-ILSVRC2013-APs.txt）。 这红色
线标记中位数AP，方框底部和顶部是第25和第75百分位数。 胡须延伸到每个的最小和最大AP
方法。 每个AP在胡须上绘制为绿点（最好以数字方式使用缩放查看）。
## 3. Visualization, ablation, and modes of error 可视化、消融、误差模式 ##
### 3.1. Visualizing learned features 可视化学习特征 ###
First-layer filters can be visualized directly第一层过滤器可以直接显示They capture oriented edges and opponent colors.它们捕捉定向边缘和相反的颜色。 

The idea is to single out a particular unit (feature) in the
network and use it as if it were an object detector in its own
right. 我们的想法是在网络中挑选出一个特征单元并使用它就好像它本身就是一个物体探测器。That is, we compute the unit’s activations on a large
set of held-out region proposals (about 10 million), sort the
proposals from highest to lowest activation, perform nonmaximum suppression, and then display the top-scoring regions.也就是说，我们在一大组候选框（大约1000万）中计算单位的激活，从最高到最低激活候选框进行排序，执行非最大抑制，然后显示最高得分区域。

![](https://i.imgur.com/tIM3Rsb.png)

Figure 4: Top regions for six pool units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts,
such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).图4：六个池化单元的顶部区域。 接收字段和激活值以白色绘制。 有些单位与概念一致，
例如人（第1行）或文本（4）。 其他单位捕获纹理和材质属性，例如点阵列（2）和镜面反射（6）。
### 3.2. Ablation studies 消融研究（消融研究通常指的是移除模型或算法的某些“特性”，并观察其如何影响性能。） ###
Performance layer-by-layer, without fine-tuning表现层，无需微调

Much of the CNN’s representational
power comes from its convolutional layers, rather than from
the much larger densely connected layers.  CNN的大部分代表性权值来自其卷积层，而不是来自密度大的层。

Performance layer-by-layer, with fine-tuning. 表现层，微调

![](https://i.imgur.com/paiul1p.png)

Table 2: Detection average precision (%) on VOC 2007 test. Rows 1-3 show R-CNN performance without fine-tuning. Rows 4-6 show
results for the CNN pre-trained on ILSVRC 2012 and then fine-tuned (FT) on VOC 2007 trainval. Row 7 includes a simple bounding-box
regression (BB) stage that reduces localization errors (Section C). Rows 8-10 present DPM methods as a strong baseline. The first uses
only HOG, while the next two use different feature learning approaches to augment or replace HOG.表2：VOC 2007测试的检测平均精度（％）。 第1-3行显示R-CNN性能而无需微调。 第4-6行显示
CNN对ILSVRC 2012进行了预训练，然后对VOC 2007 trainval进行了微调（FT）。 第7行包括一个简单的边界框
回归（BB）阶段，减少本地化错误（C部分）。 第8-10行将DPM方法作为强基线。 第一次使用
只有HOG，而接下来的两个使用不同的特征学习方法来增强或替换HOG。

Comparison to recent feature learning methods与最近的特征学习方法的比较

The first DPM feature learning method, DPM ST [28],
augments HOG features with histograms of “sketch token”
probabilities.第一个DPM特征学习方法，DPM ST [28]，
使用“草图标记”的直方图增强HOG功能
概率。

The second method, DPM HSC 第二种方法，DPM HSC
### 3.3. Network architectures 网络架构 ###
The network has a homogeneous structure
consisting of 13 layers of 3 × 3 convolution kernels, with
five max pooling layers interspersed, and topped with three
fully-connected layers.  网络具有同质结构
由13层3×3卷积核组成，含有
五个最大汇集层穿插，并加上三个
完全连接的层。We refer to this network as “O-Net”
for OxfordNet and the baseline as “T-Net” for TorontoNet.我们称这个网络为“O-Net”
对于OxfordNet和基线为TorontoNet的“T-Net”。

![](https://i.imgur.com/NNGYL6p.png)

Table 3: Detection average precision (%) on VOC 2007 test for two different CNN architectures. The first two rows are results from
Table 2 using Krizhevsky et al.’s architecture (T-Net). Rows three and four use the recently proposed 16-layer architecture from Simonyan
and Zisserman (O-Net) [43].表3：两种不同CNN架构的VOC 2007测试的检测平均精度（％）。 前两行是来自的结果
表2使用Krizhevsky等人的架构（T-Net）。 第三和第四行使用最近提出的Simonyan的16层架构
和Zisserman（O-Net）[43]。

However there
is a considerable drawback in terms of compute time, with
the forward pass of O-Net taking roughly 7 times longer
than T-Net.不过那里
在计算时间方面是一个相当大的缺点
O-Net的前进传球时间大约长7倍
比T-Net。
### 3.4. Detection error analysis 误差检测分析 ###
normalized AP规范化的AP

![](https://i.imgur.com/chIbxb5.png)

Figure 5: Distribution of top-ranked false positive (FP) types.
Each plot shows the evolving distribution of FP types as more FPs
are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with
an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion
with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [23]), significantly more of
our errors result from poor localization, rather than confusion with
background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and
the positional invariance learned from pre-training the CNN for
whole-image classification. Column three shows how our simple
bounding-box regression method fixes many localization errors.图5：排名最高的假阳性（FP）类型的分布。
每个图表显示FP类型的演变分布为更多FP
被认为是降低分数的顺序。 每个FP都被列为4种类型中的一种：Loc-poor local（一种检测方法）
IoU与0.1到0.5之间的正确等级重叠，或者重复一次）; 模拟混淆与类似的类别;OTH-混乱
具有不同的对象类别; BG-a FP在背面射击。 与DPM（见[23]）相比，显着更多
我们的错误源于糟糕的本地化，而不是混淆
背景或其他对象类，表明CNN特征比HOG更具辨别力。 宽松的本地化可能是由于我们使用自下而上的区域提案和
从CNN的预训练中学到的位置不变性
全图像分类。 第三列显示了我们的简单
边界框回归方法修复了许多本地化错误。

![](https://i.imgur.com/oHS1i0F.png)

Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [23]) for the highest and
lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part
visibility). We show plots for our method (R-CNN) with and without fine-tuning (FT) and bounding-box regression (BB) as well as for
DPM voc-release5. Overall, fine-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve
both the highest and lowest performing subsets for nearly all characteristics. This indicates that fine-tuning does more than simply improve
the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs.
Instead, fine-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility图6：对象特征的灵敏度。 每个图显示最高和最高的归一化AP（见[23]）的平均值（超过类）
六个不同对象特征中的最低性能子集（遮挡，截断，边界框区域，纵横比，视点，部分
能见度）。 我们显示了我们的方法（R-CNN）的图表，有和没有微调（FT）和边界框回归（BB）以及
DPM voc-release5。 总的来说，微调不会降低灵敏度（最大值和最小值之间的差异），但会大幅提高
几乎所有特征的最高和最低性能子集。 这表明微调不仅仅是改进
宽高比和边界框区域的性能最低的子集，可以根据我们如何扭曲网络输入来猜测。
相反，微调可以提高所有特征的鲁棒性，包括遮挡，截断，视点和零件可见性
### 3.5. Bounding-box regression 边界框回归 ###
Inspired by the
bounding-box regression employed in DPM [17], we train a
linear regression model to predict a new detection window
given the pool5 features for a selective search region proposal. Full details are given in Appendix C 灵感来自于
在DPM中使用的边界框回归[17]，我们训练了一个
线性回归模型预测新的检测窗口
鉴于pool5功能为选择性搜索区域提供了便利。 附录C中给出了完整的详细信息。
### 3.6. Qualitative results 定性结果 ###
![](https://i.imgur.com/KkMHo5H.jpg)

Figure 8: Example detections on the val2 set from the configuration that achieved 31.0% mAP on val2. Each image was sampled randomly
(these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the
precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended 图8：val2上的设置检测示例，该配置在val2上实现了31.0％mAP。 每个图像都是随机抽样的
（这些都没有策划）。 显示精度大于0.5的所有检测。 每个检测都标有预测的类和
从探测器的精确回忆曲线中检测到的精度值。 建议使用缩放以数字方式查看

![](https://i.imgur.com/NROvcYR.jpg)

Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.图9：更随机选择的示例。 有关详细信息，请参见图8标题。 建议使用缩放以数字方式查看。

![](https://i.imgur.com/2eMGPQT.jpg)

Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing
digitally with zoom is recommended.图10：策划示例。 选择每张图片是因为我们发现它令人印象深刻，令人惊讶，有趣或有趣。查看
建议使用数字缩放。

![](https://i.imgur.com/3S9rilF.jpg)

Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.图11：更多精选示例。 有关详细信息，请参见图10标题。 建议使用缩放以数字方式查看。
## 4. The ILSVRC2013 detection dataset 检测数据集 ##
### 4.1. Dataset overview 数据集概述 ###
### 4.2. Region proposals 候选区域 ###
### 4.3. Training data ###
Training data is required for three procedures in R-CNN:
(1) CNN fine-tuning, (2) detector SVM training, and (3)
bounding-box regressor training.R-CNN中的三个步骤需要训练数据：
（1）CNN微调，（2）检测器SVM训练，（3）
边界框回归训练。
### 4.4. Validation and evaluation 验证与评估 ###
### 4.5. Ablation study 消融研究 ###
![](https://i.imgur.com/yHfGX4E.png)

Table 4: ILSVRC2013 ablation study of data usage choices, fine-tuning, and bounding-box regression.表4：ILSVRC2013对数据使用选择，微调和边界框回归的消融研究。
### 4.6. Relationship to OverFeat 与OverFeat的关系 ###
OverFeat can be seen (roughly) as a special case
of R-CNN. OverFeat可以（大致）看作特殊情况
R-CNN。It is worth noting that OverFeat has
a significant speed advantage over R-CNN: it is about 9x
faster, based on a figure of 2 seconds per image quoted from
[34]. 值得注意的是OverFeat有
与R-CNN相比具有明显的速度优势：它大约是9倍
更快，基于每张图片引用的2秒数字
[34]。This speed comes from the fact that OverFeat’s sliding windows 这个速度来自于OverFeat的滑动窗口
## 5. Semantic segmentation 语义分割 ##
CNN features for segmentation CNN特征分割The first strategy (full) ignores the region’s shape and computes CNN features directly on the
warped window, exactly as we did for detection. However,
these features ignore the non-rectangular shape of the region. Two regions might have very similar bounding boxes
while having very little overlap. 第一个策略（完整）忽略区域的形状并直接计算区域的CNN特征。
扭曲的窗口，就像我们检测的那样。 然而，
这些特征忽略了区域的非矩形形状。 两个区域可能具有非常相似的边界框
虽然重叠很少。Therefore, the second strategy (fg) computes CNN features only on a region’s foreground mask. We replace the background with the mean
input so that background regions are zero after mean subtraction. 因此，第二个策略（fg）仅在区域的前地面罩上计算CNN特征。 我们用平均值替换背景
输入使得背景区域在平均次级牵引之后为零。The third strategy (full+fg) simply concatenates
the full and fg features; our experiments validate their complementarity.第三种策略（完整+ fg）简单地连接起来
完整和fg功能; 我们的实验验证了它们的实用性。

![](https://i.imgur.com/QZKBTPm.png)

Table 5: Segmentation mean accuracy (%) on VOC 2011 validation. Column 1 presents O2P; 2-7 use our CNN pre-trained on
ILSVRC 2012表5：VOC 2011评估的分段平均准确度（％）。 第1列呈现O2P; 2-7使用我们的CNN预训练
ILSVRC 2012

![](https://i.imgur.com/au3d6vO.png)

Table 6: Segmentation accuracy (%) on VOC 2011 test. We compare against two strong baselines: the “Regions and Parts” (R&P)
method of [2] and the second-order pooling (O2P) method of [4]. Without any fine-tuning, our CNN achieves top segmentation performance, outperforming R&P and roughly matching O2P.表6：VOC 2011测试的分段准确度（％）。 我们比较两个强大的基线：“地区和部分”（R＆P）
[2]的方法和[4]的二阶合并（O2P）方法。 在没有任何微调的情况下，我们的CNN实现了最高的分割性能，优于R＆P并且大致匹配O2P。
## 6. Conclusion ##
The best performing systems were complex ensembles combining multiple low-level image features with
high-level context from object detectors and scene classifiers.  性能最佳的系统是复杂的集合，将多个低级图像特征与来自对象检测器和场景分类器的高级上下文相结合。This paper presents a simple and scalable object detection algorithm 本文提出了一种简单且可扩展的物体检测算法。

 The
first is to apply high-capacity convolutional neural networks to bottom-up region proposals in order to localize
and segment objects. 第一种是将高容量卷积神经网络应用于自相而上的候选框，以便对象进行定位和分割。The second is a paradigm for training large CNNs when labeled training data is scarce. 第二个是标记的训练数据稀缺时训练大型CNNs的范式。 We
show that it is highly effective to pre-train the network—
with supervision—for a auxiliary task with abundant data
(image classification) and then to fine-tune the network for
the target task where data is scarce (detection).  我们表明，对于具有丰富数据（图像分类）的辅助任务，预先训练网络是非常有效的，然后针对数据稀缺（检测）的目标任务微调网络。 We conjecture that the “supervised pre-training/domain-specific finetuning” paradigm will be highly effective for a variety of
data-scarce vision problems.我们推测，“监督的预训练/领域特定的精细调整”范例对于各种数据缺乏的视力问题将非常有效。
## Appendix 附录 ##
### A. Object proposal transformations 候选对象转换 ###
The first method (“tightest square with context”) encloses each object proposal inside the tightest square and then scales (isotropically) the image contained in that
square to the CNN input size.第一种方法（“带有上下文的最严格的正方形”）关闭最紧凑的正方形内的每个对象提议，然后按比例缩放（各向同性地）包含在其中的图像。
平方到CNN输入大小。 The second method (“warp”)
anisotropically scales each object proposal to the CNN input size. 第二种方法（“扭曲”）
各向异性地将每个对象提议按比例缩放到CNN。

![](https://i.imgur.com/lf8GjW9.png)

Figure 7: Different object proposal transformations. (A) the
original object proposal at its actual scale relative to the transformed CNN inputs; (B) tightest square with context; (C) tightest square without context; (D) warp. Within each column and
example proposal, the top row corresponds to p = 0 pixels of context padding while the bottom row has p = 16 pixels of context
padding.图7：不同的对象提议转换。 （一）
相对于转换的CNN输入，其实际规模的原始对象提议; （B）具有背景的最严格的正方形; （C）没有背景的紧密平方; （D）翘曲。 在每列内和
示例提议，顶行对应于p = 0像素的文本填充，而底行对应p = 16像素的上下文
填充。
### B. Positive vs. negative examples and softmax 正面 VS.负面和softmax ###
没咋看
### C. Bounding-box regression 边界框回归 ###
The primary difference between the two
approaches is that here we regress from features computed
by the CNN, rather than from geometric features computed
on the inferred DPM part locations. 简单VS可变形（边界框回归）两者之间的主要区别
方法是在这里我们从计算的特征回归
由CNN，而不是从计算的几何特征
在推断的DPM部件位置。

The first is that regularization
is important:首先是正则化很重要The second issue is that care must be taken when selecting
which training pairs (P; G) to use.第二个问题是选择时必须小心
使用哪种训练对（P; G）。
### D. Additional feature visualizations 附加特征可视化 ###
![](https://i.imgur.com/CkNOxUm.png)

Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly
activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 dimensional pool5 feature map.
Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by
the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.图12：我们展示了VOC 2007测试中大约1000万个区域中最强烈的24个区域提案
激活20个单位中的每一个。 每个蒙太奇都用6×6×256维池5特征图中的单位（y，x，通道）位置标记。
每个图像区域都以白色的单位感受野的覆盖图绘制。 激活值（我们通过除以标准化
感知字段左上角显示通道中所有单位的最大激活值。 以缩放方式以数字方式观看
### E. Per-category segmentation results  每类分割结果 ###
![](https://i.imgur.com/Cn3dESL.png)

Table 7: Per-category segmentation accuracy (%) on the VOC 2011 validation set.表7：VOC 2011验证集上的每类别分割准确度（％）。
### F. Analysis of cross-dataset redundancy 交叉数据集冗余分析 ###
略...
### G. Document changelog 文档更改日志 ###

感谢：【这篇有好的解读材料一定要推荐给我0.0】