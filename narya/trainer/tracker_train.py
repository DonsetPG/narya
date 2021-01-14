from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import argparse
import os

from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

from narya.models.gluon_models import TrackerModel
from narya.datasets.tracking_dataset import TrackingDatasetBuilder

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--data_dir", default="VOCFormat/", type=str)
parser.add_argument("--backbone", default="ssd_512_resnet50_v1_coco", type=str)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--weights", default=None, type=str)
opt = parser.parse_args()

args_no_wd = True
args_label_smooth = False
args_lr_decay_period = 0
args_epochs = opt.epochs
args_warmup_epochs = 0
args_num_samples = -1
args_batch_size = opt.batch_size
args_lr = opt.lr
args_lr_mode = "step"
args_lr_decay = 0.05
args_horovod = False
args_wd = 0.0005
args_momentum = 0.9
args_amp = False
args_save_prefix = "PlayerTracker_"
args_start_epoch = 0
args_mixup = False
args_no_mixup_epochs = 20
args_log_interval = 30
args_save_interval = 10
args_val_interval = 5
args_lr_decay_epoch = "30,40,60,80,90"


try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]

print("-" * 10)
print("Building model")
print("-" * 10)

full_model = TrackerModel(pretrained=True, backbone=opt.backbone, ctx = ctx)

if opt.weights is not None:
    full_model.load_weights(opt.weights)

net = full_model.model
preprocessing_fn = full_model.preprocessing


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params("{:s}_best.params".format(prefix, epoch, current_map))
        with open(prefix + "_best_map.log", "a") as f:
            f.write("{:04d}:\t{:.4f}\n".format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params("{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(
            batch[0], ctx_list=ctx, batch_axis=0, even_split=False
        )
        label = gluon.utils.split_and_load(
            batch[1], ctx_list=ctx, batch_axis=0, even_split=False
        )
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(
                y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None
            )

        # update metric
        eval_metric.update(
            det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults
        )
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)

    if args_horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_params(),
            "sgd",
            {"learning_rate": args_lr, "wd": args_wd, "momentum": args_momentum},
        )
    else:
        trainer = gluon.Trainer(
            net.collect_params(),
            "sgd",
            {"learning_rate": args_lr, "wd": args_wd, "momentum": args_momentum},
            update_on_kvstore=(False if args_amp else None),
        )

    if args_amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args_lr_decay)
    lr_steps = sorted(
        [float(ls) for ls in args_lr_decay_epoch.split(",") if ls.strip()]
    )

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss("CrossEntropy")
    smoothl1_metric = mx.metric.Loss("SmoothL1")

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args_save_prefix + "_train.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info("Start training from [Epoch {}]".format(args_start_epoch))
    best_map = [0]

    for epoch in range(args_start_epoch, args_epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)

        for i, batch in enumerate(train_data):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # cls_targets = gluon.utils.split_and_load(batch[1][4:5], ctx_list=ctx, batch_axis=0)
            # box_targets = gluon.utils.split_and_load(batch[1][:4], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0
            )
            box_targets = gluon.utils.split_and_load(
                batch[2], ctx_list=ctx, batch_axis=0
            )

            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets
                )
                if args_amp:
                    with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)

            if not args_horovod or hvd.rank() == 0:
                local_batch_size = int(
                    args_batch_size // (hvd.size() if args_horovod else 1)
                )
                ce_metric.update(0, [l * local_batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * local_batch_size for l in box_loss])
                if args_log_interval and not (i + 1) % args_log_interval:
                    name1, loss1 = ce_metric.get()
                    name2, loss2 = smoothl1_metric.get()
                    logger.info(
                        "[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}".format(
                            epoch,
                            i,
                            args_batch_size / (time.time() - btic),
                            name1,
                            loss1,
                            name2,
                            loss2,
                        )
                    )
                btic = time.time()

        if not args_horovod or hvd.rank() == 0:
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info(
                "[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}".format(
                    epoch, (time.time() - tic), name1, loss1, name2, loss2
                )
            )
            if (epoch % args_val_interval == 0) or (
                args_save_interval and epoch % args_save_interval == 0
            ):
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
                val_msg = "\n".join(
                    ["{}={}".format(k, v) for k, v in zip(map_name, mean_ap)]
                )
                logger.info("[Epoch {}] Validation: \n{}".format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.0
            save_params(
                net, best_map, current_map, epoch, args_save_interval, args_save_prefix
            )


print("-" * 10)
print("Building dataset")
print("-" * 10)

full_dataset = TrackingDatasetBuilder(
    dataset_path=opt.data_dir,
    batch_size=opt.batch_size,
    input_shape=(512, 512),
    net=net,
)

train_dataset, val_dataset = full_dataset._get_dataset()

eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

print("length of training dataset:", len(train_dataset))
print("length of validation dataset:", len(val_dataset))

train_dataloader, valid_dataloader = full_dataset._get_dataloader()

print("-" * 10)
print("Launching the training")
print("-" * 10)

train(net, train_dataloader, valid_dataloader, eval_metric, ctx)
