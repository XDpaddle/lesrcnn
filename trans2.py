import torch
import paddle

path1 = 'weights\\lesrcnn_x3.pth'
ckpt_tr = torch.load(path1,map_location=torch.device('cpu'))
# ckpt_tr = ckpt_tr['model']
path2 = 'weights\\lesrcnn_x3.pdparams'  
ckpt_pd = paddle.load(path2)

# ------------导入模型区------------ # 导入飞桨模型，换成自己的
import config.config as option
from models import create_model
import argparse
import paddle.nn as nn

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to options YMAL file.', default="config/test/test_lesrcnn_x2.yml")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

# create model
opt_net = opt['network_G']
which_model = opt_net['which_model_G']
model = create_model(opt)
# ----------------------------------


linear_name = []
for i, (name, layer) in enumerate(model.netG.named_sublayers()):  # 记录线性层名称
    if isinstance(layer, nn.Linear): 
        print(name)
        linear_name.append(name)


# 剔除torch batchnorm2d中的num_batches_tracked参数
for key_torch, value_torch in list(ckpt_tr.items()):  # 遍历字典不能改变字典内容,需要先转成列表
    if key_torch.endswith('num_batches_tracked'):
        del ckpt_tr[key_torch]


for (key_paddle, value_paddle), (key_torch, value_torch) in zip(ckpt_pd.items(), ckpt_tr.items()):
    # print(key_paddle, key_torch)
    paddle_layernamelist = key_paddle.split('.')
    paddle_layername = ".".join(paddle_layernamelist[0:-1])  # paddle的模型层名
    if (paddle_layername in linear_name) and (paddle_layernamelist[-1]=='weight'):  # 对线性层的权重转置后再复制
        ckpt_pd[key_paddle] = paddle.to_tensor(value_torch.transpose(0,1).cpu().numpy())
    else:
        ckpt_pd[key_paddle] = paddle.to_tensor(value_torch.cpu().numpy())


    # 测试名称相等
    if key_paddle != key_torch:
        print(key_paddle, key_torch, 1)


    # 测试形状相等
    if value_paddle.numpy().shape != value_torch.cpu().numpy().shape:
        print(key_paddle, key_torch, 2, value_paddle.numpy().shape, value_torch.cpu().numpy().shape)

        if value_paddle.numpy().shape == value_torch.transpose(0,1).cpu().numpy().shape:  # torch paddle 的全连接层为转置关系  # 不对，删
            ckpt_pd[key_paddle] = paddle.to_tensor(value_torch.transpose(0,1).cpu().numpy())
            print("after transpose")
            print(key_paddle, key_torch, 2, ckpt_pd[key_paddle].numpy().shape, value_torch.cpu().numpy().shape)


paddle.save(ckpt_pd, 'trans_weights\\trans_lesrcnn_x3.pdparams')
print("OK")