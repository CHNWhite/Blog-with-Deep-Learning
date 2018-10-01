# 01、使用 SKIL 和 YOLO 构建产品级目标检测系统

在本文中，我们采用最新的神经网络实现目标检测，使用[SKIL平台](https://skymind.ai/platform)构建产品级目标检测系统。

![](https://blog.skymind.ai/content/images/2018/02/Screen-Shot-2018-02-13-at-5.22.36-PM.png)

建立一个产品级的计算机视觉系统很难，因为有很多因素需要考虑：

- 我们如何构建网络来进行预测？
- 我们以什么方式存储模型以便可以更新或回退旧版本？
- 随着客户的需求增长，我们如何提供模型预测？

除此之外，我们需要考虑在实际中使用来目标检测系统带来复杂结果的情况。

本文将引导您完成整个开发周期，并为您提供可以根据自己的目标，进行修改的程序。 它还将让您了解以下技术：

1.  SKIL的原生TensorFlow模型导入功能
2. 使用计算机视觉目标检测程序

现在让我们深入研究计算机视觉和目标检测的基础知识。

## 什么是目标检测？

计算机视觉中的目标检测可以被定义为在图像中找到具有“零到多个目标”在每张图像中。 每个对象预测都有边界框和类别概率分布。

以下是最近的三篇关于目标检测的重要论文：

1. [Faster R-CNN](https://arxiv.org/abs/1506.01497)
2. [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
3. [YOLO ("You Only Look Once") v2](https://arxiv.org/abs/1612.08242)

以前的方法处理类似任务的包括Haar Cascades，但与这些新方法相比，这种方法要慢得多。我们将重点关注下面的YOLO v2网络。

使用YOLO网络，我们将单个神经网络应用于完整图像。该网络将图像划分为区域并预测每个区域的边界框和概率。

![](https://camo.githubusercontent.com/2c54ed3d4dc4ec65d096b65c377c524c9db78876/68747470733a2f2f706a7265646469652e636f6d2f6d656469612f696d6167652f6d6f64656c322e706e67)

这些边界框由预测概率加权，其中每个对象由具有四个变量的边界框标记：对象的中心（bx，by），矩形高度（bh），矩形宽度（bw）。

我们可以从头开始训练YOLO网络，但这需要大量的工作（以及昂贵的GPU时间）。作为工程师和数据科学家，我们希望尽可能多地利用预先构建的库和机器学习模型，因此我们将使用预先训练的YOLO模型，使我们的应用程序更快，更廉价地投入生产。

# 02、使用预训练模型和SKIL的模型服务器

在之前关于[Oreilly博客的文章中](https://www.oreilly.com/ideas/integrating-convolutional-neural-networks-into-enterprise-applications)，我们讨论了如何：

*“将神经网络和卷积神经网络集成到生产的企业应用程序中本身就是一项挑战，与建模任务分开。”*

SKIL平台旨在解决那里描述的许多问题。在本文中，我们将介绍如何利用SKIL导入外部创建的TensorFlow格式模型，并使用SKIL Model Server提供预测。

![](https://blog.skymind.ai/content/images/2018/02/skil-1.0.1.png)

在这里，我们将使用在[COCO数据集](http://cocodataset.org/#home)上训练[YOLOv2](https://pjreddie.com/darknet/yolo/)模型。我们在此示例中使用的YOLO模型[设置](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg)的版本基于在COCO数据集上训练的YOLOv2体系结构。它可以识别[80个不同的类](https://github.com/pjreddie/darknet/blob/master/data/coco.names)。

权重取自以下链接，并列在YOLOv2 608x608下。

- [权重](https://pjreddie.com/media/files/yolo.weights)
- [CFG](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg)

我们采用了这个模型并将其[转换](https://github.com/thtrieu/darkflow#save-the-built-graph-to-a-protobuf-file-pb)为TensorFlow格式（protobuff，.pb），以便将其导入SKIL进行推理服务。为了使本教程更简单，我们在Github repo上[托管](https://github.com/deeplearning4j/dl4j-test-resources/blob/master/src/main/resources/tf_graphs/examples/yolov2_608x608/frozen_model.pb)了转换后的模型，供用户下载。

# 提供实时物体检测预测

机器学习从业者通常关注机器学习的建模方面，而没有充分考虑将模型投入生产所涉及的完整生命周期。在最一般的层面上，我们需要考虑机器学习建模和模型推理之间的区别; 即在模型训练后提供预测。

![](https://blog.skymind.ai/content/images/2018/02/Screen-Shot-2018-02-13-at-5.06.16-PM.png)

https://twitter.com/benhamner/status/674767904882057216

*“只有一小部分真实世界的ML系统由ML代码组成......”*

SKIL允许团队分离工作流阶段，例如建模和服务推理。SKIL还允许运营团队专注于管理横向扩展模型推理服务，而数据科学团队则专注于通过进一步培训来改进模型。在推理方面，我们有3种主要方式可以推断：

1. 传统的[OLTP](https://en.wikipedia.org/wiki/Online_transaction_processing)式单一事务推理请求通过网络（缓慢但灵活）。
2. 大规模批量推理请求（[OLAP](https://en.wikipedia.org/wiki/Online_analytical_processing)风格;例如，使用Spark在HDFS中对100万条记录进行推断，每条记录一次推断）。
3. 请求本地缓存模型的最新副本的客户端，并跳过网络遍历以在本地副本上进行多次推断。

在本教程中，我们将重点介绍最基本的推理类型，我们在网络中通过基于REST的推理请求来获取通过网络发送回远程客户端应用程序的预测。

# 03、将YOLO TensorFlow模型加载到SKIL模型服务器中

本节假设您已经[设置了](https://docs.skymind.ai/docs/releasenotes) SKIL 。如果不这样做，请查看我们的[快速入门](https://docs.skymind.ai/docs/welcome)。）
现在我们可以登录SKIL并导入[上面](https://github.com/deeplearning4j/dl4j-test-resources/blob/master/src/main/resources/tf_graphs/examples/yolov2_608x608/frozen_model.pb)提到的TensorFlow protobuff（.pb）文件。

1. 登录SKIL
2. 选择左侧工具栏上的“部署”选项
3. 单击“新部署”按钮
4. 在新创建的部署屏幕的模型部分中，选择“导入”并找到我们创建的.pb文件
5. 对于占位符选项：
   - 输入占位符的名称：“输入”（确保在输入名称后按“输入”）
   - 输出占位符的名称：“输出”（确保在输入名称后按“输入”）
6. 点击“导入模型”
7. 单击端点上的“开始”按钮

页面需要几秒钟才能报告端点已成功启动。页面将端点列为正在运行后，您将可以从页面上列出的端点访问该模型。端点URI看起来像：

```
http://localhost:9008/endpoints/tf2/model/yolo/default/
```

现在我们需要一个客户端应用程序来查询此端点并获得对象检测预测。

# 构建对象检测客户端应用程序

为了模拟一个真实的用例，我们已经包含了一个[示例客户端应用程序](https://github.com/SkymindIO/SKIL_Examples/tree/master/skil_yolo2_app)，它不仅仅是对SKIL模型服务器进行REST调用。我们在下面的代码部分中显示了SKIL客户端代码的一些关键部分。

```

        NativeImageLoader imageLoader = new NativeImageLoader(608, 608, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        INDArray imgNDArrayTmp = imageLoader.asMatrix( imgMat );
        INDArray inputFeatures = imgNDArrayTmp.permute(0, 2, 3, 1).muli(1.0 / 255.0).dup('c');

        String imgBase64 = Nd4jBase64.base64String( inputFeatures );
        Authorization auth = new Authorization();
        long start = System.nanoTime();
        String auth_token = auth.getAuthToken( "admin", "admin" );
        long end = System.nanoTime();
        System.out.println("Getting the auth token took: " + (end - start) / 1000000 + " ms");

        System.out.println( "Sending the Classification Payload..." );
        start = System.nanoTime();
        try {

            JSONObject returnJSONObject = 
                    Unirest.post( skilInferenceEndpoint + "predict" )
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .header( "Authorization", "Bearer " + auth_token)
                            .body(new JSONObject() //Using this because the field functions couldn't get translated to an acceptable json
                                    .put( "id", "some_id" )
                                    .put("prediction", new JSONObject().put("array", imgBase64))
                                    .toString())
                            .asJson()
                            .getBody().getObject(); //.toString(); 

            try {

                returnJSONObject.getJSONObject("prediction").getString("array");

            } catch (org.json.JSONException je) { 

                System.out.println( "\n\nException\n\nReturn: " + returnJSONObject );
                return;

            }

            end = System.nanoTime();
            System.out.println("SKIL inference REST round trip took: " + (end - start) / 1000000 + " ms");


            String predict_return_array = returnJSONObject.getJSONObject("prediction").getString("array");
            System.out.println( "REST payload return length: " + predict_return_array.length() );
            INDArray networkOutput = Nd4jBase64.fromBase64( predict_return_array );
```

此示例的SKIL [客户端代码](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java)将执行以下任务：

1. 使用SKIL进行[身份验证](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java#L248)并获取令牌
2. [Base64编码](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java#L245)我们想要预测的图像
3. 获取auth令牌和base64图像字节，[并通过REST将它们发送](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java#L259)到SKIL进行推理
4. [Base64解码](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java#L288)从SKIL模型服务器返回的结果
5. 应用TensorFlow模型所需的后推理[激活函数](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java#L360)（通过YoloUtils类）（特别是）
6. 在原始图像上渲染输出[边界框](https://github.com/SkymindIO/SKIL_Examples/blob/master/skil_yolo2_app/client_app/src/main/java/ai/skymind/skil/examples/yolo2/modelserver/inference/YOLO2_TF_Client.java#L365)，如下所示

![](https://blog.skymind.ai/content/images/2018/02/Screen-Shot-2018-02-13-at-5.08.16-PM.png)

对于SKIL模型服务器中托管的普通DL4J和Keras模型，我们不必应用后推理激活函数。但是，TensorFlow网络不会自动将激活功能应用于最终层。要完成此示例，我们必须使用提供的`YoloUtils`类方法在客户端代码中应用这些激活函数。

使用以下命令[克隆](https://github.com/SkymindIO/SKIL_Examples/tree/master/skil_yolo2_app#run-the-skil-client-locally-with-the-sample-client-application)此[repo](https://github.com/SkymindIO/SKIL_Examples/tree/master/skil_yolo2_app)以获取包含的YOLOv2示例应用程序，该应用程序将检索预测并在本地呈现边界框：

```
git clone git@github.com:SkymindIO/SKIL_Examples.git
```

然后我们需要专门构建YOLOv2客户端应用程序JAR文件：

```
cd skil_yolo2_app/client_app
mvn -U package
```

这将构建一个`skil-example-yolo2-tf-1.0.0.jar`在`./target`子目录的`client_app/`子目录中命名的JAR文件。

现在我们有了一个客户端应用程序JAR，我们可以从命令行运行yolo2客户端JAR：

```
java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input [image URI] --endpoint [SKIL Endpoint URI]
```

说明

- `--input` 可以是您选择的任何输入图像（带有file：//前缀的本地文件，或带有http：//前缀的Internet URI的图像文件）
- `--endpoint` parameter是导入TF .pb文件时创建的端点

使用此命令的一个示例是：

```
java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/images/0012.jpg --endpoint http://localhost:9008/endpoints/tf2/model/yolo/default/
```

这个repo将构建一个名为“skil-example-yolo2-tf-1.0.0.jar”的JAR，以便我们可以从命令行运行yolo2客户端JAR：

```
java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input [image URI] --endpoint [SKIL Endpoint URI]
```

此客户端应用程序将允许我们获取任何图像的预测，并在图像上呈现边界框+分类以及在上图中看到。

# 04、总结和未来的想法

YOLO演示非常有趣，可以发送您自己的图像以查看它可以选择的内容。如果要引用本地文件系统上的文件，只需在URI中替换`http://`，`file://`如下例所示：

```
java -jar ./target/skil-example-yolo2-tf-1.0.0.jar --input file:///tmp/beach.png --endpoint [SKIL Endpoint URI]
```

你会看到YOLO非常善于挑选出微妙的物体，正如我们在下面复杂的街景中所看到的那样。

![截屏，2018年1月22日，在-5.34.55-PM](https://blog.skymind.ai/content/images/2018/02/Screen-Shot-2018-01-22-at-5.34.55-PM.png)

要了解有关YOLO如何工作的更多信息以及您可以在SKIL上使用它构建的其他内容，请查看以下资源：

- 理解对象检测中的边界框机制（又名“理解YOLO输出”）
  - <http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html>
- 针对特定用例进一步培训（专业）YOLO的更多示例：
  - <https://github.com/experiencor/basic-yolo-keras>
- 利用Tiny-YOLO网络构建视频检测系统
  - <http://ramok.tech/2018/01/18/java-autonomous-driving-car-detection/>