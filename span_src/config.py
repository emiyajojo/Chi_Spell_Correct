import argparse

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser.add_argument('--raw_data_dir', default='/hy-tmp/data', help='数据集的存放路径')
        parser.add_argument('--output_dir', default='/hy-tmp/src/SimCSE-out', help='训练好的模型输出路径')
        parser.add_argument('--bert_dir', default='/hy-tmp/bert', help='可以支持 ernie, roberta-wwm, bert, macbert')
        parser.add_argument('--task_type', default='span', help='NER任务采用的模型类别, crf/span')
        parser.add_argument('--loss_type', default='ls_ce', help='损失函数的类型, crf/span')
        # other args
        parser.add_argument('--seed', type=int, default=2023, help='随机种子, random seed')
        parser.add_argument('--gpu_ids', type=str, default=['0'], help='GPU的id信息, "-1"代表cpu, "0,1,..."代表GPU')
        parser.add_argument('--mode', type=str, default='train', help='当前模式, 训练或推理')
        # train args
        parser.add_argument('--train_epochs', default=10, type=int, help='训练模型的轮次数')
        parser.add_argument('--dropout_prob', default=0.1, type=float, help='dropout比例')
        parser.add_argument('--lr', default=2e-5, type=float, help='学习率, 针对BERT系列')
        parser.add_argument('--other_lr', default=2e-3, type=float, help='学习率, 针对BERT外的模块')
        parser.add_argument('--max_grad_norm', default=1.0, type=float, help='最大梯度裁剪值')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--train_batch_size', default=64, type=int)
        parser.add_argument('--test_file', default='')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()