藏文历史文献数字化系统
=========================
##环境搭建
###
1. 环境搭建 安装 [Anaconda3](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive)
2. 配置运行env
```bash
    $ conda create -n digit_2 python=3.5
    $ source activate digit_2
    $ conda install matplotlib scikit-learn scikit-image hdf5
    $ pip install tensorflow
    $ pip install keras
    $ pip install h5py
```
### 包版本信息
scikit-image 0.13.0  
matplotlib 2.0.2  
scikit-learn 0.19.0  
numpy 1.13.1  
tensorflow 1.4.0  
keras 2.1.3  
h5py 2.7.1     
hdf5 1.8.17  


##文档结构
* root  
   * ui  对应于ui文件的窗体类，用于各种事件的添加  
       * auto_ui pyqt5的UI文件，只用于设计UI和自动生成，**请勿手动修改.ui生成的.py文件**   
   * [run.py](run.py)  使用该文件启动程序
   * [process_batch.py](process_batch.py) 文件 使用该文件将经过噪点去除的。
   生成文本并进行初步的归类 归类的方法是按照 余弦相似度
   * [words_Titan.txt](words_Titan.txt) 生成的模型的分类对应的Unicode码值
   * [cnn_encoder.py](cnn_encoder.py) 生成cnn自编码器的文件，
   利用process_batch中生成的数据集文件data.npz,进行更精确的数据分类。
   
## 执行步骤
若要运行程序，则直接
```bash
python run.py
```
即可


若要从新开始运行数据并训练则需要根据以下步骤执行
1. 准备数据
    1. 使用[process_batch.py](process_batch.py) ```batch_extract_char_2_file``` 
    方法切分数据到指定目录
    2. 使用 [process_batch.py](process_batch.py) 余弦相似度方法 ```classify_image_by_cosine_similarity```
    初步分类数据
    3. 手动标记数据将相似的文件归并到同一个目录中,目录的编号对应该字符的Unicode码，子弟采用FZZW_QT.TTF
    ，使用 [cnn_encoder.py](cnn_encoder.py) 对文件进行编码，生成模型
    4. 使用最终的分类好的数据
2. 训练数据
    1. 运行环境 python-2.7 opencv linux keras
    2. 运行方法  
        1. 运行 ```select_sample.py```:从对应的文件夹选择一定数目的样本,用于抽取一定数目的样本和划分训练集和测试集.  
        2. 运行 ```data_augument.py```:数据增加方法,旋转,缩放和增加噪音.
        3. 运行 ```lenet5_v2.py``` 训练并保存数据模型
        


==============
## 更新记录
2018-05-07 更新了字符模型的加载方式，提升了运行速度