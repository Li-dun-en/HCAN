import torch
import numpy as np

class Group_helper(object):
    def __init__(self, dataset, num_groups, scale_y=None, Max = None, Min = None):
        '''
            # dataset : list of deltas (CoRe method) or list of scores (RT method)
            dataset : 2D array-like of shape (N, C), where N is the number of samples, and C is the number of features.
            depth : depth of the tree
            Symmetrical: (bool) Whether the group is symmetrical about 0.
                        if symmetrical, dataset only contains th delta bigger than zero.
            Max : maximum score or delta for a certain sports.
        '''
        self.num_features = dataset.shape[1]  # C
        self.datasets = [sorted(dataset[:, c]) for c in range(self.num_features)]     # 对每个特征的数据进行排序,这是为了通过分位数来划分边界,保证每个类别的样本数大致平衡
        self.lengths = [len(d) for d in self.datasets]
        self.num_leaf = num_groups
        self.scale = scale_y    #保存缩放器
        self.max = Max if Max is not None else [d[-1] for d in self.datasets]
        self.min = Min if Min is not None else [d[0] for d in self.datasets]        #算出绝对边界
        self.Groups = [[] for _ in range(self.num_features)]
        # print("self.max, self.min:", self.max, self.min)
        self.build()

    def build(self):
        '''
            separate region of each leaf
            # Now, group_helper.Groups contains the groups for each feature
        '''
        for c in range(self.num_features):
            dataset = self.datasets[c]  #取出第c个特征的所有数据
            length = self.lengths[c]    #取出该特征的数据长度
            #取出特征的最大和最小值
            max_val = self.max[c]       
            min_val = self.min[c]
            for i in range(self.num_leaf):                                          #计算切分点的索引
                Region_left = dataset[int((i / self.num_leaf) * (length - 1))]
                Region_left = min_val if i == 0 else Region_left                # 如果i是第一个分组，则Region_left等于最小值，否则取出分位数对应的值

                Region_right = dataset[int(((i + 1) / self.num_leaf) * (length - 1))]
                Region_right = max_val if i == self.num_leaf - 1 else Region_right

                self.Groups[c].append([Region_left, Region_right])   #将计算出的[左边界, 右边界]存入到Groups中
        # print('Groups:', self.Groups)# , self.scale.inverse_transform(self.Groups[c]), self.scale.inverse_transform(self.Groups[c]))

    def produce_label(self, scores):
        # scores: array-like of shape [N, C]
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        glabels = []  # Group labels for classification         #存分类标签(给UAC用)
        rlabels = []  # Regression labels within each group     #存放偏移量(给L_REG用)

        # -----第一层循环，遍历每个特征通道-----

        for c in range(self.num_features):  # Iterate over each feature
            feature_scores = scores[:, c]
            feature_glabels = []
            feature_rlabels = []

            # -----第二层循环，遍历每一个时间步的数值-----
            for score in feature_scores:
                #假设分为4类
                # leaf_glabels: 初始化为[0,0,0,0] (One-hot)
                # leaf_rlabels: 初始化为[-1,-1,-1,-1] (默认值-1代表无效)
                leaf_glabels = [0] * self.num_leaf
                leaf_rlabels = [-1] * self.num_leaf
                # print("score, feature_scores:", score, feature_scores)
                
                # -------第三层循环: 遍历之前划定好的4个区间-------
                for i, group in enumerate(self.Groups[c]):
                    # print("i, group:", i, group)
                    if group[0] <= score < group[1]:
                        leaf_glabels[i] = 1
                        #制作回归标签(偏移量):
                        #leaf_rlabels 变成[-1,0.5,-1.-1]
                        leaf_rlabels[i] = (score - group[0]) / (group[1] - group[0])
                        # print("group[0], score, group[1], leaf_glabels, leaf_rlabels, i:", group[0], score, group[1], leaf_glabels, leaf_rlabels, i)

                feature_glabels.append(leaf_glabels)
                feature_rlabels.append(leaf_rlabels)
                # print("leaf_glabels, leaf_rlabels:", leaf_glabels, leaf_rlabels)

            glabels.append(feature_glabels)
            rlabels.append(feature_rlabels)

        #形状: [特征数C, 样本数N，类别数K]
        glabels = torch.tensor(glabels, dtype=torch.float32)
        rlabels = torch.tensor(rlabels, dtype=torch.float32)
        #生成整数索引标签(用于CrossEntropyLoss)
        #argmax 把[0,1,0,0]变成索引1
        cls_labels = torch.argmax(glabels, dim=-1)

        # print(cls_labels.shape, glabels.shape)  # cls_label输出应为 (feature, N) glabels输出为 (feature, N, num_class)
        return cls_labels, glabels, rlabels

    def inference(self, probs, deltas):
        '''
            # probs: bs * leaf      #模型预测的分类概率
            # delta: bs * leaf      #模型预测的回归偏移量
            probs: array-like of shape [N, C, num_leaf]
            deltas: array-like of shape [N, C, num_leaf]
        '''
        predictions = []            #存放最终还原出的预测结果

        #遍历每一个样本
        for n in range(probs.shape[0]):
            sample_predictions = []
            #遍历每一个特征通道
            for c in range(self.num_features):
                #取出该样本该特征的分类概率和回归偏移量
                prob = probs[n, c] #概率向量，比如 [0.1, 0.8, 0.05, 0.05]
                delta = deltas[n, c] #偏移量向量，比如 [0.2, 0.5, 0.3, 0.4]

                #找组
                leaf_id = prob.argmax()
                #找到第leaf_id个组的边界
                group = self.Groups[c][leaf_id]

                ## 特殊情况处理：如果左边界等于右边界（单个值的区间），直接取值
                if group[0] == group[1]:
                    prediction = group[0] + delta[leaf_id]
                else:
                    #核心公式: 预测值 = 左边界 + (区间宽度 * 偏移量)
                    prediction = group[0] + (group[1] - group[0]) * delta[leaf_id]

                sample_predictions.append(prediction)
            predictions.append(sample_predictions)
            
        return predictions
        # return torch.tensor(predictions, dtype=torch.float32)

    def get_Group(self):
        return self.Groups

    def number_leaf(self):
        return self.num_leaf


def main():
    # Example usage
    np.random.seed(0)  # 为了可重复性
    N, C = 5, 3  # N个样本，C个特征
    scores = np.random.rand(N, C) * 10  # 随机分数，范围从0到10

    # 初始化Group_helper
    num_groups = 2  # 每个特征的分组数量
    group_helper = Group_helper(scores, num_groups)

    # 生成标签
    ont_hot, glabels, rlabels = group_helper.produce_label(torch.from_numpy(scores))

    print("分组的所有数据：")
    print(scores)
    # 显示前几个样本的标签
    print("One-hot分类标签:")
    print(ont_hot)
    print("分类标签 (前5个样本):")
    print(glabels)
    print("\n回归标签 (前5个样本):")
    print(rlabels)

    # 假设的模型预测概率和回归值
    # 这通常来自于你的模型，这里我们用随机值来模拟
    probs = torch.rand(N, C, num_groups)
    deltas = torch.rand(N, C, num_groups) * 10  # 假设回归值范围也是0到10

    # 使用inference进行推断
    predictions = group_helper.inference(probs, deltas)

    print("所有probs：")
    print(probs)
    print("所有deltas")
    print(deltas)
    # 显示前几个样本的预测结果
    print("\n预测结果 (前5个样本):")
    print(predictions)

    exit()
    RT_depth = 4
    # normalize(dataset[i][2], class_idx, score_range)
    score_range = 100
    delta_list = [20.3, 23.4, 10.5, 21.5, 36.7]
    num_leaf = 2 ** (RT_depth - 1)
    print(num_leaf)
    group = Group_helper(delta_list, RT_depth, Symmetrical=True, Max=score_range, Min=0)
    glabel_1, rlabel_1 = group.produce_label([20, 34, 35])
    print(glabel_1, rlabel_1)
    '''
    tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 1]], device='cuda:0') tensor([[-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [ 0.9852, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000],
        [-1.0000,  0.1384,  0.1514]], device='cuda:0')
    竖着看是一个数字的label
    '''
    leaf_probs_2 = [[2.3, 3.6], [6.3, 3.6], [6.6, 3.6], [2.6, 3.6], [3.6, 3.6], [3.2, 3.6], [6.2, 3.6], [7.1, 3.6]]
    leaf_probs_2 = torch.tensor(leaf_probs_2)
    delta_2 = [[2.2, 3.6], [3.5, 3.6], [6.7, 3.6], [2.4, 3.6], [1.3, 3.6], [2.6, 3.6], [1.4, 3.6], [7.4, 3.6]]  # [batch_size]，每个b_s里面表示叶子节点的个数，区间的个数
    delta_2 = torch.tensor(delta_2)
    print(leaf_probs_2.shape, delta_2.shape)    # torch.Size([8, 2]) torch.Size([8, 2])
    relative_scores = group.inference(leaf_probs_2, delta_2)
    print(relative_scores)

if __name__ == '__main__':
    main()