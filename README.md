# BFGAN
Generating Behavior Features for Cold-Start Spam Review Detection
1、review_shuffle_w2v_c1w8-i20h0n5s100.txt为利用word2vec对文本信息训练的预训练模型
2、trainEmb为所有标签的序号/train、test为带标签的
3、bf_embedding为每个用户的六种行为特征为RBFs
4、rd_embedding/time_embbding为离散化特征即EAFs
5、textConv为通过textcnn训练的文本特征

3/4/5都只给了一条样例数据

pinjie文件夹为最后的三种行为特征的拼接分类结果
