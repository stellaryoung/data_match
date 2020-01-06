import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# name是加上标题行
# 通过df["标题名"]可以直接访问那一列的信息
train_df = pd.read_table("sms_train.txt", names=["label", "sms_message"])
test_df = pd.read_table("sms_test.txt", names=["test_message"])

# 标签信息数字化表示
# 将标签中spam与ham全部转化为对应的数字1和0,1代表是垃圾信息，0代表不是
train_df["label"] = train_df["label"].map({'ham': 0, 'spam': 1})   # map中是字典，给出映射关系
# 将训练集的标签单独拿出来
trainLabel = train_df.pop("label").values

# 英语句子数字化表示
# 这里采用的是词袋模型来表示句子特征
countVector = CountVectorizer()
# fit对文本建立词汇表，transform将文本中的句子转化为词频向量
# fit_transform返回的采用稀疏表示的特征矩阵
featureVectorSparseMatrix = countVector.fit_transform(train_df["sms_message"])
# 将稀疏表示的特征矩阵转换为二维numpy数组
# print(type(featureVectorSparseMatrix.toarray()))
featureMatrix = featureVectorSparseMatrix.toarray()
print(featureMatrix.shape)    # 特征矩阵的维度(3448, 6797)
print(trainLabel.shape)       # 样本标签数量(3448,)

# 将需要测试的信息文本也都用词袋模型特征向量化,注意测试集不需要fit,否则得到的词频向量长度会有差异
testFeatureMatrix = countVector.transform(test_df["test_message"]).toarray()
print(testFeatureMatrix.shape)
'''-------------------------------------------------------------------------------------------'''
# 经过预处理处理形成的结果数据：
# featureMatrix:特征矩阵
# testFeatureMatrix:测试信息文本的特征矩阵
# trainLabel:标签数据的numpy一维数组

# 采用多项式朴素贝叶斯模型，该模型专用于处理文本数据
feature = featureMatrix
label = trainLabel
test = testFeatureMatrix
model = MultinomialNB()
model.fit(feature, label)  # 训练模型

res = model.predict(test)
submission_df = pd.DataFrame(data={'Id': test_df.index+1, 'Res': res})

print('--------------------------res:输出分类结果文件-------------------------------------')
print(submission_df.head())
submission_df.to_csv('res.csv', sep=",", header=False, index=False)