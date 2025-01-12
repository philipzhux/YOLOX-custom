import os
from yolox.exp import Exp as MyExp
from torch.utils.tensorboard import SummaryWriter

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # Number of classes
        self.num_classes = 2

        # Model size settings
        self.depth = 0.33
        self.width = 0.50

        # Input sizes resized to this somehow
        self.input_size = (640, 640)  # (height, width)
        self.random_size = (14, 26)
        self.test_size = (640, 640)

        # Dataset paths
        self.data_dir = "dataset"
        self.train_ann = os.path.join(self.data_dir, "train.txt")
        self.val_ann = os.path.join(self.data_dir, "val.txt")

        # Training parameters
        self.max_epoch = 50
        self.print_interval = 10
        self.eval_interval = 10

        # Data loader settings
        self.data_num_workers = 4
        self.batch_size = 8

        # Initialize a TensorBoard writer
        # You can customize the log directory as needed
        self.tensorboard_logdir = "./tensorboard_logs"
        self.writer = SummaryWriter(log_dir=self.tensorboard_logdir)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import YoloDataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler
        from yolox.data import worker_init_reset_seed

        dataset = YoloDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=0.5,
                hsv_prob=1.0,
            ),
            cache=False,
        )

        sampler = InfiniteSampler(
            len(dataset), seed=self.seed if self.seed else 0
        )
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_reset_seed,
        }

        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )
        return dataloader

    def get_eval_loader(self, batch_size, is_distributed):
        from yolox.data import YoloDataset, ValTransform
        from torch.utils.data import DataLoader

        valdataset = YoloDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name='val',
            img_size=self.test_size,
            preproc=ValTransform(),
            cache=False,
        )

        dataloader = DataLoader(
            valdataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )
        return dataloader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import VOCEvaluator  # Using VOC evaluator
        val_loader = self.get_eval_loader(batch_size, is_distributed)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=0.01,
            nmsthre=0.45,
            num_classes=self.num_classes,
        )
        return evaluator

    def log_metrics(self, metrics: dict, step: int):
        """
        Args:
            metrics (dict): A dictionary of {metric_name: metric_value}
            step (int): The current step or epoch number
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close_tensorboard(self):
        """
        Close the TensorBoard writer once training is complete.
        """
        self.writer.close()
