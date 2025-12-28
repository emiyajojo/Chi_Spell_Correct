import os
import copy
import torch

# swa 滑动平均模型, ⼀般在训练平稳阶段再使⽤SWA
def swa(model, model_dir):
    model_path_list = get_model_path_list(model_dir)
    swa_model = copy.deepcopy(model)
    swa_n = 0.
    with torch.no_grad():
        for _ckpt in model_path_list:
            print('Load model from {}'.format(_ckpt))
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())
            alpha = 1. / (swa_n + 1.)
            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1
    return swa_model

# 从⽂件夹中获取 model.pt 的路径
def get_model_path_list(base_dir):
    tmp = os.listdir(base_dir)
    # 文档示例逻辑：获取epoch 60到80的模型
    tmp = ['sup_epoch_{}.pt'.format(i) for i in range(60, 80) if 'sup_epoch_{}.pt'.format(i) in tmp]
    model_lists = [os.path.join(base_dir,x) for x in tmp]
    return model_lists

# 文档中引用了FGM和PGD，但未提供具体实现代码，此处仅做占位说明
class FGM:
    def __init__(self, model):
        self.model = model
        pass

class PGD:
    pass