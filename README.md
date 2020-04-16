# BFGAN
Generating Behavior Features for Cold-Start Spam Review Detection（2019DASFAA）（2020Information Sciences）
http://link.springer.com/chapter/10.1007/978-3-030-18590-9_38（DASFAA）
https://www.sciencedirect.com/science/article/pii/S0020025520302437?dgcid=author（Information Sciences）

The implementation of the model in the paper.


1）review_shuffle_w2v_c1w8-i20h0n5s100.txt  is a pre training model for text information trained by word2vec.

2) trainEmb are indexs of all examples while train、test are indexs of examples with labels.

3）bf_embedding are six RBFs of users'.

4）rd_embedding/time_embbding are EAFs of users'.

5）textConv are text features trained by textcnn. 

3/4/5 only give an example.

In pinjie, there is the final result of spliced RBFs.

# Cite
If the codes help you, please cite our paper:
[1]Xiaoya Tang,Tieyun Qian,Zhenni You. Generating Behavior Features for Cold-Start Spam Review Detection[J]. 2019DASFAA.

[1]Xiaoya Tang,Tieyun Qian,Zhenni You. Generating Behavior Features for Cold-Start Spam Review Detection with Adversarial Learning[J]. Information Sciences,2020.

