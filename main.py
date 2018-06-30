from config import opt
import os
import torch as t
import models
from data import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm


def train(**kwargs):
    # 从配置文件获取opt，将参数赋值到对应的kwargs
    opt.parse(kwargs)
    # 指定env=opt.env,默认端口为8097,host为localhost
    vis = Visualizer(opt.env)

    # 步骤1：模型参数
    # 获取models中的opt.model属性值，决定采用何种网络
    # model为网络结构，注意等式右边的括号，如果没有括号，则调用网络需要使用model()
    model = getattr(models, opt.model)()
    # 如果有预训练好的网络，则采用之
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 步骤2：加载数据
    # 加载训练集数据
    # 使用自定义数据集类DogCat加载数据，使用train_data.__getitem__(index)获取对应索引的data和label
    train_data = DogCat(opt.train_data_root, train=True)
    # 使用PyTorch自带的utils.data中的DataLoader加载数据
    train_dataloader = DataLoader(
        train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # 加载验证集数据
    val_data = DogCat(opt.train_data_root, train=False)
    # DataLoader是一个可迭代对象，可将dataset返回的每条数据拼接成一个batch，并提供多线程加速优化，当所有数据遍历完一次，对DataLoader完成一次迭代
    # DataLoader提供对数据分批，打乱，使用多少子进程加载数据
    val_dataloader = DataLoader(
        val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 步骤3：损失函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    # 建立优化器，指定要调整的参数和学习率
    optimizer = t.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=opt.weight_decay)

    # 步骤4：meters
    loss_meter = meter.AverageValueMeter()        # 用于统计任意添加的方差和均值，可以用来测量平均损失
    # 多类之间的混淆矩阵，初始化时，指定类别数为2,参数normalized指定是否归一化名，归一化后输出百分比，否则输出数值
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):
        # 重置统计信息
        loss_meter.reset()
        confusion_matrix.reset()

        # tqdm:Python的进度条工具，输入为一个迭代器
        # enumerate：对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        # enumerate多用于在for循环中得到计数
        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            # 训练模型

            # 定义输入输出变量
            input = Variable(data)
            target = Variable(label)
            # 是否使用gpu
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            # 所有参数的梯度清零
            optimizer.zero_grad()
            # 获取得分
            score = model(input)
            # 计算损失
            loss = criterion(score, target)
            # 反向传播
            loss.backward()
            # 更新网络的权重和参数
            optimizer.step()

            # 参数更新和可视化
            loss_meter.add(loss.data[0])  # 将loss加入loss_meter计算平均损失
            confusion_matrix.add(score.data, target.data)

            # 判断是否是第 N 个batch，以决定要不要打印信息
            if ii % opt.print_freq == opt.print_freq-1:
                vis.plot('loss', loss_meter.value()[0])      # 绘制loss曲线

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        # 保存模型至checkpoints文件夹
        model.save()

        # 验证和可视化
        # 获取验证集混淆矩阵和验证集准确率
        val_cm, val_accuracy = val(model, val_dataloader)

        # 绘制验证集准确率曲线
        vis.plot('val_accuracy', val_accuracy)
        # 输出基本信息，如loss，学习率，混淆矩阵等
        vis.log("epoch:{epoch},\n lr:{lr},\n loss:{loss},\n train_cm:{train_cm},\n val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()), lr=lr
        ))

        # 更新学习率
        # 若当前损失比之前的要高，则认为学习率过快，需要降速
        if loss_meter.value()[0] > previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                # 更新优化器中的学习率
                param_group['lr'] = lr

        # 更新previous_loss
        previous_loss = loss_meter.value()[0]


def val(model, dataloader):

    # 将模型设置为验证模式，之后还需要重新设置为训练模式，这部分会影响BatchNorm和Dropout等层的运行
    model.eval()

    # 初始化混淆矩阵为二分类
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
        # 计算验证集得分
        score = model(val_input)
        # 计算混淆矩阵
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    # 将模型设置回训练模式
    model.train()

    # 获取混淆矩阵
    cm_value = confusion_matrix.value()
    # 计算正确分类的准确率
    accuracy = 100.*(cm_value[0][0]+cm_value[1][1])/(cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    opt.parse(kwargs)

    # 建立模型，使用验证模式
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        # 模型置入GPU
        model.cuda()

    # 获取数据
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(
        test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            # 数据置入GPU
            input = input.cuda()
        # 计算测试集得分
        score = model(input)
        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        if os.path.exists(opt.debug_file):
            import ipdb
            ipdb.set_trace()
        # zip：将多个迭代器部分组合成一个元组，然后返回由元组组成的列表
        batch_results = [(path_, probability_)
                         for path_, probability_ in zip(path, probability)]
        results += batch_results
    # 将结果写入csv文件，文件名由opt.result_file指定
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == '__main__':
    import fire
    fire.Fire()
