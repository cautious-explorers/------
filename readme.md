题目：[讯飞社交账号网络分类挑战赛](http://challenge.xfyun.cn/topic/info?type=social-media-account&option=ssgy):利用无向图的连接数据，确定图中节点的类别。

解决方法：
1. best solutions:本地十折交叉acc.8662，样本外acc为.8682
- node2vec确定节点embedding
- 使用高斯判别器
2. 已尝试方法但不是最佳：
- 使用中心度等原始图节点特征作为embedding （结果最差在.74左右）
- 使用GAT、SAGE、GCN （结果波动较大，最大在.861左右）
- 得到节点embedding之后使用常用的机器学习方法分类（xgboost、svm、lr、lightgbm等，最好的是catboost能达到.866）


复现：
run train.sh in the code file: **bash train.sh** 


环境：
cuda 11.4
包信息在requirements.txt中

