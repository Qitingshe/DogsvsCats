import warnings

class DefaultConfig(object):
    env='default'   # visdom环境
    model='ResNet34'

    train_data_root='./data/train'
    test_data_root='./data/test1'
    load_model_path=None

    batch_size=32
    use_gpu=True
    num_workers=4   # n 个子进程用于加载数据
    print_freq=20   # 每 N batch 打印一次信息

    debug_file='/tmp/debug'
    result_file='result.csv'

    max_epoch=30
    lr=0.0001
    lr_decay=0.5
    weight_decay=0e-5

def parse(self,kwargs):
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self,k,v)
    
    # 打印用户配置
    print('user config/用户配置:')
    # 将所有属性字典列出，并打印
    for k,v in self.__class__.__dict__.items():
        # 排除私有属性
        if not k.startswith('__'):
            print(k,getattr(self,k))

DefaultConfig.parse=parse
opt=DefaultConfig()