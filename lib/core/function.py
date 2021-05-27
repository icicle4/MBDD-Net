# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch
import cv2
import json

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.utils import save_posetrack_json
from collections import defaultdict
from tqdm import tqdm

from poseval.eval_helpers import Joint
from poseval.evaluateAP import evaluateAP
from poseval import eval_helpers
from poseval.evaluateTracking import evaluateTracking

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def finetune_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        # output = model(input)[:, :3]
        # target = target.cuda(non_blocking=True)[:, :3]

        target_weight = target_weight.cuda(non_blocking=True)
        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def inference(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda()
            output = model(input)
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            #debug_vis_in_raw_image(meta, preds)

def debug_vis_in_raw_image(meta, preds, c, s):
    image_paths = meta["image"]

    kp_num = preds.shape[1]
    for i, image_path in enumerate(image_paths):
        if image_path == target_image_path:
            print('final_pred', preds[i])
            print('i', i)
            print('center', c[i])
            print('scale', s[i])
            img = cv2.imread(image_path)

            for k in range(kp_num):
                img = cv2.circle(img, (int(preds[i, k, 0]), int(preds[i, k, 1])), 4, (0, 255, 0))

            cv2.imshow('debug', img)
            cv2.waitKey(0)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda()
            output = model(input)
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def center_of_two_point(kp1, kp2):
    return ((kp1[0] + kp2[0]) / 2, (kp1[1] + kp2[1]) / 2)


def openpose_to_posetrack(keypoints):
    posetrack_points = []
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    posetrack_points.append(keypoints[0])
    posetrack_points.append(center_of_two_point(left_eye, right_eye))
    posetrack_points.append(center_of_two_point(left_ear, right_ear))
    posetrack_points.extend(keypoints[5:])

    for i, point in enumerate(posetrack_points):
        point.update({"id": [i]})
    return posetrack_points


def compute_posetrack_metrics(gt_path, out_posetrack_path, outputDir):
    argv = []
    argv.append("multi")
    argv.append(gt_path)
    argv.append(out_posetrack_path)
    gtFramesAll, prFramesAll = eval_helpers.load_data_dir(argv)
    print("# gt frames  :", len(gtFramesAll))
    print("# pred frames:", len(prFramesAll))

    # compute AP
    print("Evaluation of per-frame multi-person pose estimation")
    apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll, outputDir, True, True)

    # print AP
    print("Average Precision (AP) metric:")
    eval_helpers.printTable(apAll)

    return apAll

target_image_path = "data/posetrack/images/val/001007_mpii_test/000000.jpg"

def inference_pose_track(config, val_loader, val_dataset, model, criterion, output_dir,
                        val_annotation_root):
    model.eval()
    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda()
            output = model(input)

            if target_image_path not in meta["image"]:
                continue

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            debug_vis_in_raw_image(meta, preds, c, s)



def validate_pose_track(config, val_loader, val_dataset, model, criterion, output_dir,
                        val_annotation_root):
    # switch to evaluate mode
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    all_seq_dataset = defaultdict(list)
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda()
            output = model(input)

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            #loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            #losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            num_images = input.size(0)
            keypoint_num = output.size(1)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            image_file_names = meta['image']
            seq_names = meta['seq_name']
            image_ids = [x.numpy() for x in meta['image_id']]

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

            for i in range(num_images):
                image_name = image_file_names[i]

                # x, y, w, h = [bboxs[idx][i] for idx in range(4)]
                # bbox_head = [bbox_heads[idx][i] for idx in range(4)]
                image_id = image_ids[i]
                seq_name = seq_names[i]
                keypoints = []
                scores = []

                for k in range(keypoint_num):
                    keypoints.extend([float(preds[i, k, 0]), float(preds[i, k, 1]),  float(maxvals[i, k, 0])])
                    scores.append(float(maxvals[i, k, 0]))

                # down_neck = center_of_two_point([float(preds[i, 5, 0]), float(preds[i, 5, 1])], [float(preds[i, 6, 0]), float(preds[i, 6, 1])])
                # head_bottom = center_of_two_point(down_neck, [float(preds[i, 0, 0]), float(preds[i, 0, 1])])
                # center_eye = center_of_two_point([float(preds[i, 1, 0]), float(preds[i, 1, 1])], [float(preds[i, 2, 0]), float(preds[i, 2, 1])])
                # nose = [float(preds[i, 0, 0]), float(preds[i, 0, 1])]
                # diff = (center_eye[0] - nose[0], center_eye[1] - nose[1])
                # head_top = (nose[0] - diff[0], nose[1], diff[1])
                #
                # keypoints[3] = head_bottom[0]
                # keypoints[4] = head_bottom[1]
                #
                # keypoints[6] = head_top[0]
                # keypoints[7] = head_top[1]

                keypoints = keypoints[:9] + [0, 0, 0, 0, 0, 0] + keypoints[15:]
                scores = scores[:3] + [0, 0] + scores[5:]

                all_seq_dataset[seq_name].append(
                    {
                        "track_id": i,
                        "image_id": int(image_id),
                        "keypoints": keypoints,
                        "scores": scores
                    }
                )

    for seq_name in all_seq_dataset.keys():
        save_json_path = os.path.join(output_dir, "pred", "{}.json".format(seq_name))
        raw_json_path = os.path.join(val_annotation_root, "{}.json".format(seq_name))

        with open(raw_json_path, 'r') as f:
            content = json.load(f)

        pred_content = content.copy()
        pred_content["annotations"] = all_seq_dataset[seq_name]

        with open(save_json_path, 'w') as f:
            json.dump(pred_content, f)

    apAll = compute_posetrack_metrics(val_annotation_root, os.path.join(output_dir, "pred/"),
                                      os.path.join(output_dir, "metrics/"))
    total = apAll[-1][0]
    return total


def _validate_pose_track(config, val_loader, val_dataset, model):
    # switch to evaluate mode
    model.eval()

    annorects = []

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(tqdm(val_loader)):
            # compute output
            output = model(input)
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)
            keypoint_num = output.size(1)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            image_file_names = meta['image']
            bboxs = [x.numpy() for x in meta['bbox']]
            bbox_heads = [x.numpy() for x in meta['bbox_head']]
            image_ids = [x.numpy() for x in meta['image_id']]

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            for i in range(num_images):
                image_name = image_file_names[i]

                # x, y, w, h = [bboxs[idx][i] for idx in range(4)]
                # bbox_head = [bbox_heads[idx][i] for idx in range(4)]
                image_id = image_ids[i]

                keypoints = []
                scores = []

                for k in range(keypoint_num):
                    keypoints.extend([float(preds[i, k, 0]), float(preds[i, k, 1]),  float(maxvals[i, k, 0])])
                    scores.append(float(maxvals[i, k, 0]))

                down_neck = center_of_two_point([float(preds[i, 5, 0]), float(preds[i, 5, 1])], [float(preds[i, 6, 0]), float(preds[i, 6, 1])])
                head_bottom = center_of_two_point(down_neck, [float(preds[i, 0, 0]), float(preds[i, 0, 1])])
                center_eye = center_of_two_point([float(preds[i, 1, 0]), float(preds[i, 1, 1])], [float(preds[i, 2, 0]), float(preds[i, 2, 1])])
                nose = [float(preds[i, 0, 0]), float(preds[i, 0, 1])]
                diff = (center_eye[0] - nose[0], center_eye[1] - nose[1])
                head_top = (nose[0] - diff[0], nose[1], diff[1])

                keypoints[3] = head_bottom[0]
                keypoints[4] = head_bottom[1]

                keypoints[6] = head_top[0]
                keypoints[7] = head_top[1]

                annorects.append(
                    {
                        "track_id": i,
                        "image_id": int(image_id),
                        "keypoints": keypoints,
                        "scores": scores
                    }
                )
    return annorects

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
