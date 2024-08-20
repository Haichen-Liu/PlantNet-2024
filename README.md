# PlantNet-Deep-Plant-Species-Recognition

Hello, if you 're interested in using this code for plant identification, take a look at the steps to use this code. 

First of all, create a new folder ( total folder ), the project files are downloaded to this folder;

Secondly, create a new folder and name the new folder as ' dataset ';

Then, the two compression packages of ' dataset-train ' and ' dataset-test ' are decompressed into the new ' dataset ' folder;

Next, modify the naming of these two folders, that is, ' dataset-train ' to ' train ', ' dataset-test ' to ' test ' ( of course, you can also modify the code to read the file path in the programming software ) ;

Finally, the use of the code steps:

First, run grayscale.py ( generate images for image processing );

Secondly,  run train.py ( build your own model, noting that the model can vary depending on your computer 's performance );

Finally, run predict.py ( realize the function of plant recognition ). 

In addition, if you want to generate a picture of the confusion matrix, or get the value of Precision, Recall, F1-score, you can write the relevant code in train.py to get them.

您好，如果您有兴趣使用这套代码进行植物识别，可以看看这套代码的使用步骤。

首先，新建一个文件夹（总文件夹），把项目里的文件都下载到这个文件夹里面；

其次，再新建一个文件夹，并命名这个新建的文件夹为"dataset";

然后，将"dataset-train"和"dataset-test"两个压缩包解压到新建的"dataset"文件夹中;

再然后，修改这两个文件夹的命名，即将"dataset-train"改成"train",将"dataset-test"改成"test"（当然你也可以在编程软件里修改读取文件路径的代码）。

最后，是代码的使用步骤：

首先，运行binarization grayscale.py（生成图像处理的图片）,

其次，运行train.py（生成自己的模型，需要注意的是，模型可以会由于电脑性能的区别而不同）,

最后，运行predict.py（实现植物识别的功能）。

此外，如果你想生成混淆矩阵的图片，或者得到Precision,Recall,F1-score的值，可以在train.py中编写相关代码来获得。

