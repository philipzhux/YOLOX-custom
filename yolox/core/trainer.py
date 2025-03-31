#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
import json
import fcntl
from contextlib import contextmanager
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    MlflowLogger,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    bboxes_iou,
    cxcywh2xyxy
)


@contextmanager
def file_lock(path):
    """Context manager for file locking"""
    with open(path, 'r+') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def load_json_atomic(path):
    """Load JSON file with file locking"""
    if not os.path.exists(path):
        return {}
    with file_lock(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {path}: {e}")
            return {}

def save_json_atomic(path, data):
    """Save JSON file with file locking"""
    with file_lock(path) as f:
        json.dump(data, f, indent=2)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

def update_json_atomic(path, updates):
    """Update JSON file with file locking"""
    with file_lock(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
        data.update(updates)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        # Check environment variable "COMPUTE_TPR"
        self.compute_tpr_flag = True
        # self.compute_tpr_flag = (os.getenv("COMPUTE_TPR", "0") == "1")

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        # Get paths from environment
        self.config_path = os.getenv("YOLOX_CONFIG_PATH")
        self.state_path = os.getenv("YOLOX_STATE_PATH")
        self.dynamic_config_enabled = os.getenv("YOLOX_ENABLE_DYNAMIC_CONFIG") == "1"
        
        if self.dynamic_config_enabled:
            logger.info("Dynamic configuration enabled")
            logger.info(f"Config path: {self.config_path}")
            logger.info(f"State path: {self.state_path}")

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error("Exception in training: ", e)
            if self.dynamic_config_enabled:
                try:
                    update_json_atomic(self.state_path, {"running": False})
                    update_json_atomic(self.config_path, {"running": False})
                except Exception as state_e:
                    logger.error(f"Failed to update state on error: {state_e}")
            raise
        finally:
            self.after_train()
            if self.dynamic_config_enabled:
                try:
                    update_json_atomic(self.state_path, {"running": False})
                    update_json_atomic(self.config_path, {"running": False})
                except Exception as e:
                    logger.error(f"Failed to update final state: {e}")

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        tpr_value = None
        if self.compute_tpr_flag:
            # temporarily switch to eval so model returns decoded boxes
            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_training):
                batch_predictions = self.model(inps)
            self.model.train()
            tpr_value = self.compute_tpr_stub(batch_predictions, targets)
        iter_end_time = time.time()

        # optionally log tpr value
        if tpr_value is not None:
            outputs["tpr"] = tpr_value
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))
        
        update_json_atomic(self.state_path, {"running": True})

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
                self.mlflow_logger.on_train_end(self.args, file_name=self.file_name,
                                                metadata=metadata)

    def before_epoch(self):
        """Apply any pending configuration changes at epoch boundary"""
        if not self.dynamic_config_enabled:
            return
            
        try:
            config = load_json_atomic(self.config_path)
                
            # Check batch size change
            new_batch_size = config.get("batch_size")
            if new_batch_size and int(new_batch_size) != int(self.args.batch_size):
                new_batch_size = int(new_batch_size)
                logger.info(f"Applying batch size change from {self.args.batch_size} to {new_batch_size}")
                self.exp.basic_lr_per_img = self.exp.basic_lr_per_img * self.args.batch_size / new_batch_size
                self.args.batch_size = new_batch_size
                # Recreate dataloader
                self.train_loader = self.exp.get_data_loader(
                    batch_size=self.args.batch_size,
                    is_distributed=self.is_distributed,
                    no_aug=self.no_aug,
                    cache_img=self.args.cache,
                )
                self.prefetcher = DataPrefetcher(self.train_loader)
                self.max_iter = len(self.train_loader)
                
                # Update learning rate scheduler
                new_lr = config.get("learning_rate")
                if not new_lr:
                    new_lr = self.exp.basic_lr_per_img * self.args.batch_size
                self.lr_scheduler = self.exp.get_lr_scheduler(new_lr, self.max_iter)
                        
            # Update training state
            updates = {
                "batch_size": self.args.batch_size,
                "learning_rate": new_lr
            }
            update_json_atomic(self.config_path, updates)
            update_json_atomic(self.state_path, updates)
                
        except Exception as e:
            logger.error(f"Error checking config changes: {e}")
        
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
                if self.args.logger == 'mlflow':
                    logs = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    logs.update({"train/lr": self.meter["lr"].latest})
                    self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)

            self.meter.clear_meters()

            if self.dynamic_config_enabled and (self.iter + 1) % self.exp.print_interval == 0:
                try:
                    loss_meter = self.meter.get_filtered_meter("loss")
                    updates = {
                        "epoch": self.epoch + 1,
                        "iteration": self.iter + 1,
                        "learning_rate_adaptive": self.lr_scheduler.lr,
                        "batch_size": self.args.batch_size,
                    }
                    for k, v in loss_meter.items():
                        if v.latest is not None:
                            updates[f"{k}"] = float(v.latest)
                    updates.update(
                        {
                            "progress": progress_str,
                            "mem": mem_str,
                            "time": time_str,
                            "eta": eta_str,
                        }
                    )
                    update_json_atomic(self.state_path, updates)
                except Exception as e:
                    logger.error(f"Failed to update training state: {e}")

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                # Base metrics
                self.tblogger.add_scalar("performance/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("performance/COCOAP50_95", ap50_95, self.epoch + 1)
                
                if type(summary) is dict and "apr" in summary:
                    recall_by_class = summary["apr"]
                    precision_by_class = summary.get("precision", {})
                    tp_fp_metrics = summary.get("tp_fp_metrics", {})
                    
                    # Track metrics for fairness calculation
                    recalls = []
                    precisions = []
                    
                    # Calculate distributions for bias amplification
                    gt_counts = {}
                    pred_counts = {}
                    total_gt = 0
                    total_pred = 0
                    
                    for cls_name in recall_by_class:
                        # Get base metrics
                        recall = recall_by_class[cls_name]
                        precision = precision_by_class.get(cls_name, 0.0)
                        recalls.append(recall)
                        precisions.append(precision)
                        
                        # Get TP, FP, FN counts
                        cls_metrics = tp_fp_metrics.get(cls_name, {})
                        tp = cls_metrics.get("true_positives", 0)
                        fp = cls_metrics.get("false_positives", 0)
                        fn = cls_metrics.get("false_negatives", 0)
                        
                        # For bias amplification calculation
                        gt_count = tp + fn
                        pred_count = tp + fp
                        gt_counts[cls_name] = gt_count
                        pred_counts[cls_name] = pred_count
                        total_gt += gt_count
                        total_pred += pred_count
                        
                        # Calculate metrics
                        total = tp + fp + fn
                        accuracy = tp / total if total > 0 else 0.0
                        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
                        
                        # 1. Raw Counts
                        self.tblogger.add_scalar(f"raw/tp/{cls_name}", tp, self.epoch + 1)
                        self.tblogger.add_scalar(f"raw/fp/{cls_name}", fp, self.epoch + 1)
                        self.tblogger.add_scalar(f"raw/fn/{cls_name}", fn, self.epoch + 1)
                        self.tblogger.add_scalar(f"raw/total/{cls_name}", total, self.epoch + 1)
                        
                        # 2. Performance Metrics
                        self.tblogger.add_scalar(f"performance/accuracy/{cls_name}", accuracy, self.epoch + 1)
                        self.tblogger.add_scalar(f"performance/f1/{cls_name}", f1_score, self.epoch + 1)
                        
                        # 3. TPR (True Positive Rate) related
                        self.tblogger.add_scalar(f"tpr/recall/{cls_name}", recall, self.epoch + 1)
                        self.tblogger.add_scalar(f"tpr/precision/{cls_name}", precision, self.epoch + 1)
                        
                        # 4. Distribution and Bias Amplification
                        gt_ratio = gt_count / total_gt if total_gt > 0 else 0
                        pred_ratio = pred_count / total_pred if total_pred > 0 else 0
                        bias_amp = pred_ratio / gt_ratio if gt_ratio > 0 else float('inf')
                        
                        self.tblogger.add_scalar(f"distribution/gt_ratio/{cls_name}", gt_ratio, self.epoch + 1)
                        self.tblogger.add_scalar(f"distribution/pred_ratio/{cls_name}", pred_ratio, self.epoch + 1)
                        self.tblogger.add_scalar(f"bias/amplification/{cls_name}", bias_amp, self.epoch + 1)
                    
                    # 5. Fairness Metrics
                    if recalls:
                        min_recall = min(recalls)
                        max_recall = max(recalls)
                        avg_recall = sum(recalls) / len(recalls)
                        recall_fairness = min_recall / (max_recall + 1e-6)
                        
                        min_precision = min(precisions)
                        max_precision = max(precisions)
                        avg_precision = sum(precisions) / len(precisions)
                        precision_fairness = min_precision / (max_precision + 1e-6)
                        
                        # Group fairness metrics
                        self.tblogger.add_scalar("fairness/recall/min", min_recall, self.epoch + 1)
                        self.tblogger.add_scalar("fairness/recall/max", max_recall, self.epoch + 1)
                        self.tblogger.add_scalar("fairness/recall/avg", avg_recall, self.epoch + 1)
                        self.tblogger.add_scalar("fairness/recall/ratio", recall_fairness, self.epoch + 1)
                        
                        self.tblogger.add_scalar("fairness/precision/min", min_precision, self.epoch + 1)
                        self.tblogger.add_scalar("fairness/precision/max", max_precision, self.epoch + 1)
                        self.tblogger.add_scalar("fairness/precision/avg", avg_precision, self.epoch + 1)
                        self.tblogger.add_scalar("fairness/precision/ratio", precision_fairness, self.epoch + 1)
                        
                        # Calculate overall bias metrics
                        bias_amps = [abs(amp - 1) for amp in [pred_counts[cls] / total_pred / (gt_counts[cls] / total_gt) 
                                   for cls in gt_counts] if gt_counts[cls] > 0]
                        if bias_amps:
                            max_bias = max(bias_amps)
                            avg_bias = sum(bias_amps) / len(bias_amps)
                            self.tblogger.add_scalar("bias/max_amplification", max_bias, self.epoch + 1)
                            self.tblogger.add_scalar("bias/avg_amplification", avg_bias, self.epoch + 1)

                        logger.info(
                            f"\nEpoch {self.epoch + 1} Metrics:"
                            f"\nFairness - Recall Min: {min_recall:.3f}, Max: {max_recall:.3f}, Ratio: {recall_fairness:.3f}"
                            f"\nFairness - Precision Min: {min_precision:.3f}, Max: {max_precision:.3f}, Ratio: {precision_fairness:.3f}"
                            f"\nBias - Max Amplification: {max_bias:.3f}, Avg Amplification: {avg_bias:.3f}"
                        )

            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/best_ap": round(self.best_ap, 3),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)

            logger.info("\n" + summary["info"] if (type(summary) is dict and "info" in summary) else str(summary))
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        if self.args.logger == "mlflow":
            metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
            self.mlflow_logger.save_checkpoints(self.args, self.exp, self.file_name, self.epoch,
                                                metadata, update_best_ckpt)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            # Update state file with new checkpoint
            if self.dynamic_config_enabled:
                try:
                    checkpoint_path = os.path.join(self.file_name, ckpt_name)
                    update_json_atomic(self.state_path, {
                        "last_checkpoint": checkpoint_path
                    })
                    logger.info(f"Updated state with new checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to update state with checkpoint: {e}")

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )



    def compute_tpr_stub(self, predictions, targets, iou_thresh=0.5, conf_thresh=0.5):
        """
        Compute batch-level TPR given YOLOX-style decoded predictions and GT targets.

        Args:
            predictions (Tensor): shape (B, N, 5 + num_classes)
                For each batch element b:
                    predictions[b, :, :4]  = [cx, cy, w, h]
                    predictions[b, :, 4]   = obj_conf
                    predictions[b, :, 5:]  = cls_conf for each class
            targets (Tensor): shape (B, M, 5)
                For each batch element b:
                    targets[b, m, 0] = class_id
                    targets[b, m, 1:] = [cx, cy, w, h]
            iou_thresh (float): minimum IoU to count as TP
            conf_thresh (float): minimum confidence to filter predictions

        Returns:
            float: TPR = TP / (TP + FN) across the entire batch.
        """
        device = predictions.device
        batch_size = predictions.shape[0]
        total_tp = 0
        total_fn = 0

        for b in range(batch_size):
            # ------------------------------------------
            # (1) Get the predictions for image b
            # ------------------------------------------
            preds_b = predictions[b]  # shape: (N, 5 + num_classes)
            if preds_b.numel() == 0:
                # if no predictions at all
                gt_b = targets[b]  # shape: (M, 5)
                num_gts_b = (gt_b[:, 1:].sum(dim=1) > 0).sum().item()  # or just M if your data is well-formed
                total_fn += num_gts_b
                continue

            # Separate out boxes and confidences
            # 4 coords: [cx, cy, w, h]
            pred_bboxes_cxcywh = preds_b[:, 0:4]
            obj_conf = preds_b[:, 4]  # shape: (N,)

            # If you want class-agnostic confidence:
            if preds_b.shape[1] > 5:
                cls_conf = preds_b[:, 5:]  # shape: (N, num_classes)
                max_cls_conf, _ = cls_conf.max(dim=-1)  # shape: (N,)
                final_conf = obj_conf * max_cls_conf
            else:
                # If there's no separate class scores, just use obj_conf:
                final_conf = obj_conf

            # ------------------------------------------
            # (2) Filter out low-confidence predictions
            # ------------------------------------------
            keep_mask = final_conf >= conf_thresh
            if keep_mask.sum() == 0:
                # All preds are below confidence threshold
                gt_b = targets[b]
                num_gts_b = (gt_b[:, 1:].sum(dim=1) > 0).sum().item()
                total_fn += num_gts_b
                continue

            pred_bboxes_cxcywh = pred_bboxes_cxcywh[keep_mask]
            final_conf = final_conf[keep_mask]

            # ------------------------------------------
            # (3) Convert predicted bboxes from cx,cy,w,h -> x1,y1,x2,y2
            # ------------------------------------------
            pred_bboxes_xyxy = cxcywh2xyxy(pred_bboxes_cxcywh)

            # ------------------------------------------
            # (4) Prepare GT bboxes
            # ------------------------------------------
            gt_b = targets[b]  # shape (M, 5) => [class_id, cx, cy, w, h]
            # Filter out any zero row if your dataset can have empty boxes
            # Otherwise, assume all are valid if M>0
            # Let's assume M is the number of valid GT for this image:
            # Or YOLOX often uses nlabel to find how many are valid
            valid_gt_mask = (gt_b[:, 1:].sum(dim=1) > 0)
            gt_b = gt_b[valid_gt_mask]
            if gt_b.numel() == 0:
                # No GT => no TPs, no FNs
                continue

            gt_bboxes_cxcywh = gt_b[:, 1:5]
            gt_bboxes_xyxy = cxcywh2xyxy(gt_bboxes_cxcywh)

            # ------------------------------------------
            # (5) Match GT boxes with predicted boxes
            # ------------------------------------------
            # We'll do a "greedy" approach: for each GT box,
            # if there's any pred box with IoU >= iou_thresh, we call it a TP.
            # (You could do more advanced matching if desired.)
            ious = bboxes_iou(gt_bboxes_xyxy, pred_bboxes_xyxy, xyxy=True)
            # ious: shape (n_gt, n_pred)

            # For each GT, see if any pred hits iou_thresh
            max_ious, _ = ious.max(dim=1)  # shape (n_gt,)
            # A GT is "matched" if max IoU >= iou_thresh
            matched = (max_ious >= iou_thresh)
            tp_b = matched.sum().item()
            fn_b = (matched == False).sum().item()

            total_tp += tp_b
            total_fn += fn_b

        # ------------------------------------------
        # (6) Compute TPR across the entire batch
        # ------------------------------------------
        if (total_tp + total_fn) == 0:
            # e.g. batch had no GT objects
            return 0.0
        tpr = total_tp / (total_tp + total_fn)
        return float(tpr)