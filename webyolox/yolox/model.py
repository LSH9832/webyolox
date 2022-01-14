from yolox.exp import Exp as MyExp
import yaml


class Exp(MyExp):
    def __init__(
            self,
            exp_name,
            max_epoch=300,
            output_dir="/path/to/your/output_dir",
            data_dir="/path/to/your/dataset",
            train_dir="/path/to/your/train_dir",
            val_dir="/path/to/your/val_dir",
            train_ann="/path/to/your/train_json",
            val_ann="/path/to/your/val_json",
    ):
        super(Exp, self).__init__()

        self.exp_name = exp_name
        if self.exp_name == 'yolox-s':
            self.depth = 0.33
            self.width = 0.50
        elif self.exp_name == 'yolox-m':
            self.depth = 0.67
            self.width = 0.75
        elif self.exp_name == 'yolox-l':
            self.depth = 1.0
            self.width = 1.0
        elif self.exp_name == 'yolox-x':
            self.depth = 1.33
            self.width = 1.25
        elif self.exp_name == 'yolox-tiny':
            self.depth = 0.33
            self.width = 0.375
            self.input_scale = (416, 416)
            self.mosaic_scale = (0.5, 1.5)
            self.random_size = (10, 20)
            self.test_size = (416, 416)
            self.enable_mixup = False

        self.output_dir = output_dir
        self.data_dir, self.train_dir, self.val_dir = data_dir, train_dir, val_dir
        self.train_ann, self.val_ann = train_ann, val_ann
        self.max_epoch = max_epoch

    def load_yaml(self, yaml_name):
        yaml_data = yaml.load(open(yaml_name, 'r'), yaml.FullLoader)

        self.warmup_epochs = yaml_data['warmup_epochs']

        self.warmup_lr = yaml_data['warmup_lr']
        self.basic_lr_per_img = yaml_data['basic_lr_per_img']
        self.scheduler = yaml_data['scheduler']
        self.no_aug_epochs = yaml_data['no_aug_epochs']
        self.min_lr_ratio = yaml_data['min_lr_ratio']
        self.ema = yaml_data['ema']

        self.weight_decay = yaml_data['weight_decay']
        self.momentum = yaml_data['momentum']
        self.print_interval = yaml_data['print_interval']
        self.eval_interval = yaml_data['eval_interval']
