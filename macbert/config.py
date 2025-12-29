import argparse

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser.add_argument('--bert_dir', default='./chi_macbert', help='the dir of bert weights')
        parser.add_argument('--model_name', default='bert4csc', help='model for macbert4csc')
        parser.add_argument('--hyper_params', default=0.2, help='params for model')
        parser.add_argument('--train_data', default='./data/train.json', help='train data for model')
        parser.add_argument('--valid_data', default='./data/dev.json', help='valid data for model')
        parser.add_argument('--test_data', default='./data/dev.json', help='test data for model')
        parser.add_argument('--predict_dev_path', default='./output/predict_dev_file.txt', help='predict result of dev')
        parser.add_argument('--res_path', default='./output/predict_dev_correct_detail.txt', help='correct detail of dev')
        parser.add_argument('--seed', type=int, default=2023, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default=['0'], help='-1 for cpu, 0, 1 for multi gpu')
        parser.add_argument('--train_epochs', default=10, type=int, help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.1, type=float, help='drop out probability')
        parser.add_argument('--mode', default='train', type=str, help='train or test')
        parser.add_argument('--lr', default=2.5e-5, type=float, help='learning rate for the bert module')
        parser.add_argument('--other_lr', default=2.5e-3, type=float, help='lr for the other module')
        parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--output_dir', default='output/bert4csc', type=str)
        parser.add_argument('--use_fp16', default=False, action='store_true', help='weather to use fp16 during training')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
