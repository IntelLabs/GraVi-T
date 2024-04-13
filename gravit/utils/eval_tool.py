# This code is based the official ActivityNet repository: https://github.com/activitynet/ActivityNet
# The owner of the official ActivityNet repository: ActivityNet
# Copyright (c) 2015 ActivityNet
# Licensed under The MIT License
# Please refer to https://github.com/activitynet/ActivityNet/blob/master/LICENSE

import os
import numpy as np
import pandas as pd

from collections import defaultdict
import csv
import decimal
import heapq
import h5py
from .ava import object_detection_evaluation
from .ava import standard_fields
from .vs import knapsack
from scipy import stats
from scipy.stats import rankdata

def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != float or recall.dtype != float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Smooth precision to be monotonically decreasing.
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def load_csv(filename, column_names):
  """Loads CSV from the filename using given column names.
  Adds uid column.
  Args:
    filename: Path to the CSV file to load.
    column_names: A list of column names for the data.
  Returns:
    df: A Pandas DataFrame containing the data.
  """
  # Here and elsewhere, df indicates a DataFrame variable.
  df = pd.read_csv(filename, header=None, names=column_names)
  # Creates a unique id from frame timestamp and entity id.
  df["uid"] = (df["frame_timestamp"].map(str) + ":" + df["entity_id"])
  return df


def eq(a, b, tolerance=1e-09):
  """Returns true if values are approximately equal."""
  return abs(a - b) <= tolerance


def merge_groundtruth_and_predictions(df_groundtruth, df_predictions):
  """Merges groundtruth and prediction DataFrames.
  The returned DataFrame is merged on uid field and sorted in descending order
  by score field. Bounding boxes are checked to make sure they match between
  groundtruth and predictions.
  Args:
    df_groundtruth: A DataFrame with groundtruth data.
    df_predictions: A DataFrame with predictions data.
  Returns:
    df_merged: A merged DataFrame, with rows matched on uid column.
  """
  if df_groundtruth["uid"].count() != df_predictions["uid"].count():
    raise ValueError(
        "Groundtruth and predictions CSV must have the same number of "
        "unique rows.")

  if df_predictions["label"].unique() != ["SPEAKING_AUDIBLE"]:
    raise ValueError(
        "Predictions CSV must contain only SPEAKING_AUDIBLE label.")

  if df_predictions["score"].count() < df_predictions["uid"].count():
    raise ValueError("Predictions CSV must contain score value for every row.")

  # Merges groundtruth and predictions on uid, validates that uid is unique
  # in both frames, and sorts the resulting frame by the predictions score.
  df_merged = df_groundtruth.merge(
      df_predictions,
      on="uid",
      suffixes=("_groundtruth", "_prediction"),
      validate="1:1").sort_values(
          by=["score"], ascending=False).reset_index()
  # Validates that bounding boxes in ground truth and predictions match for the
  # same uids.
  df_merged["bounding_box_correct"] = np.where(
      eq(df_merged["entity_box_x1_groundtruth"],
         df_merged["entity_box_x1_prediction"])
      & eq(df_merged["entity_box_x2_groundtruth"],
           df_merged["entity_box_x2_prediction"])
      & eq(df_merged["entity_box_y1_groundtruth"],
           df_merged["entity_box_y1_prediction"])
      & eq(df_merged["entity_box_y2_groundtruth"],
           df_merged["entity_box_y2_prediction"]), True, False)

  if (~df_merged["bounding_box_correct"]).sum() > 0:
    raise ValueError(
        "Mismatch between groundtruth and predictions bounding boxes found at "
        + str(list(df_merged[~df_merged["bounding_box_correct"]]["uid"])))

  return df_merged


def get_all_positives(df_merged):
  """Counts all positive examples in the groundtruth dataset."""
  return df_merged[df_merged["label_groundtruth"] ==
                   "SPEAKING_AUDIBLE"]["uid"].count()


def calculate_precision_recall(df_merged):
  """Calculates precision and recall arrays going through df_merged row-wise."""
  all_positives = get_all_positives(df_merged)

  # Populates each row with 1 if this row is a true positive
  # (at its score level).
  df_merged["is_tp"] = np.where(
      (df_merged["label_groundtruth"] == "SPEAKING_AUDIBLE") &
      (df_merged["label_prediction"] == "SPEAKING_AUDIBLE"), 1, 0)

  # Counts true positives up to and including that row.
  df_merged["tp"] = df_merged["is_tp"].cumsum()

  # Calculates precision for every row counting true positives up to
  # and including that row over the index (1-based) of that row.
  df_merged["precision"] = df_merged["tp"] / (df_merged.index + 1)

  # Calculates recall for every row counting true positives up to
  # and including that row over all positives in the groundtruth dataset.
  df_merged["recall"] = df_merged["tp"] / all_positives

  return np.array(df_merged["precision"]), np.array(df_merged["recall"])


def run_evaluation_asd(predictions, groundtruth):
  """Runs AVA Active Speaker evaluation, returns average precision result."""
  column_names=[
      "video_id", "frame_timestamp", "entity_box_x1", "entity_box_y1",
      "entity_box_x2", "entity_box_y2", "label", "entity_id"
  ]
  df_groundtruth = load_csv(groundtruth, column_names=column_names)
  df_predictions = pd.DataFrame(predictions, columns=column_names+["score"])
  # Creates a unique id from frame timestamp and entity id.
  df_predictions["uid"] = (df_predictions["frame_timestamp"].map(str) + ":" + df_predictions["entity_id"])

  df_merged = merge_groundtruth_and_predictions(df_groundtruth, df_predictions)
  precision, recall = calculate_precision_recall(df_merged)

  return compute_average_precision(precision, recall)


def make_image_key(video_id, timestamp):
  """Returns a unique identifier for a video id & timestamp."""
  return "%s,%.6f" % (video_id, decimal.Decimal(timestamp))


def read_csv(csv_file, class_whitelist=None, capacity=0):
  """Loads boxes and class labels from a CSV file in the AVA format.
  CSV file format described at https://research.google.com/ava/download.html.
  Args:
    csv_file: A file object.
    class_whitelist: If provided, boxes corresponding to (integer) class labels
      not in this set are skipped.
    capacity: Maximum number of labeled boxes allowed for each example. Default
      is 0 where there is no limit.
  Returns:
    boxes: A dictionary mapping each unique image key (string) to a list of
      boxes, given as coordinates [y1, x1, y2, x2].
    labels: A dictionary mapping each unique image key (string) to a list of
      integer class lables, matching the corresponding box in `boxes`.
    scores: A dictionary mapping each unique image key (string) to a list of
      score values lables, matching the corresponding label in `labels`. If
      scores are not provided in the csv, then they will default to 1.0.
    all_keys: A set of all image keys found in the csv file.
  """
  entries = defaultdict(list)
  boxes = defaultdict(list)
  labels = defaultdict(list)
  scores = defaultdict(list)
  all_keys = set()
  reader = csv.reader(csv_file)
  for row in reader:
    assert len(row) in [2, 7, 8], "Wrong number of columns: " + row
    image_key = make_image_key(row[0], row[1])
    all_keys.add(image_key)
    # Rows with 2 tokens (videoid,timestatmp) indicates images with no detected
    # / ground truth actions boxes. Add them to all_keys, so we can score
    # appropriately, but otherwise skip the box creation steps.
    if len(row) == 2:
      continue
    x1, y1, x2, y2 = [float(n) for n in row[2:6]]
    action_id = int(row[6])
    if class_whitelist and action_id not in class_whitelist:
      continue
    score = 1.0
    if len(row) == 8:
      score = float(row[7])
    if capacity < 1 or len(entries[image_key]) < capacity:
      heapq.heappush(entries[image_key], (score, action_id, y1, x1, y2, x2))
    elif score > entries[image_key][0][0]:
      heapq.heapreplace(entries[image_key], (score, action_id, y1, x1, y2, x2))
  for image_key in entries:
    # Evaluation API assumes boxes with descending scores
    entry = sorted(entries[image_key], key=lambda tup: -tup[0])
    for item in entry:
      score, action_id, y1, x1, y2, x2 = item
      boxes[image_key].append([y1, x1, y2, x2])
      labels[image_key].append(action_id)
      scores[image_key].append(score)
  return boxes, labels, scores, all_keys


def read_detections(detections, class_whitelist, capacity=50):
  """
  Loads boxes and class labels from a list of detections in the AVA format.
  """
  entries = defaultdict(list)
  boxes = defaultdict(list)
  labels = defaultdict(list)
  scores = defaultdict(list)
  for row in detections:
    image_key = make_image_key(row[0], row[1])
    x1, y1, x2, y2 = row[2:6]
    action_id = int(row[6])
    if class_whitelist and action_id not in class_whitelist:
      continue
    score = float(row[7])
    if capacity < 1 or len(entries[image_key]) < capacity:
      heapq.heappush(entries[image_key], (score, action_id, y1, x1, y2, x2))
    elif score > entries[image_key][0][0]:
      heapq.heapreplace(entries[image_key], (score, action_id, y1, x1, y2, x2))
  for image_key in entries:
    # Evaluation API assumes boxes with descending scores
    entry = sorted(entries[image_key], key=lambda tup: -tup[0])
    for item in entry:
      score, action_id, y1, x1, y2, x2 = item
      boxes[image_key].append([y1, x1, y2, x2])
      labels[image_key].append(action_id)
      scores[image_key].append(score)
  return boxes, labels, scores


def read_labelmap(labelmap_file):
  """Reads a labelmap without the dependency on protocol buffers.
  Args:
    labelmap_file: A file object containing a label map protocol buffer.
  Returns:
    labelmap: The label map in the form used by the object_detection_evaluation
      module - a list of {"id": integer, "name": classname } dicts.
    class_ids: A set containing all of the valid class id integers.
  """
  labelmap = []
  class_ids = set()
  name = ""
  class_id = ""
  for line in labelmap_file:
    if line.startswith("  name:"):
      name = line.split('"')[1]
    elif line.startswith("  id:") or line.startswith("  label_id:"):
      class_id = int(line.strip().split(" ")[-1])
      labelmap.append({"id": class_id, "name": name})
      class_ids.add(class_id)
  return labelmap, class_ids


def run_evaluation_al(detections, groundtruth, labelmap):
  """
  Runs AVA Actions evaluation, returns mean average precision result
  """
  with open(labelmap, 'r') as f:
    categories, class_whitelist = read_labelmap(f)

  pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

  # Reads the ground truth data.
  with open(groundtruth, 'r') as f:
    boxes, labels, _, included_keys = read_csv(f, class_whitelist)
  for image_key in boxes:
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array(boxes[image_key], dtype=float),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array(labels[image_key], dtype=int),
            standard_fields.InputDataFields.groundtruth_difficult:
                np.zeros(len(boxes[image_key]), dtype=bool)
        })

  # Reads detections data.
  boxes, labels, scores = read_detections(detections, class_whitelist)
  for image_key in boxes:
    if image_key not in included_keys:
      continue
    pascal_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                np.array(boxes[image_key], dtype=float),
            standard_fields.DetectionResultFields.detection_classes:
                np.array(labels[image_key], dtype=int),
            standard_fields.DetectionResultFields.detection_scores:
                np.array(scores[image_key], dtype=float)
        })

  metrics = pascal_evaluator.evaluate()
  return metrics['PascalBoxes_Precision/mAP@0.5IOU']


def get_class_start_end_times(result):
    """
    Return the classes and their corresponding start and end times
    """
    last_class = result[0]
    classes = [last_class]
    starts = [0]
    ends = []

    for i, c in enumerate(result):
        if c != last_class:
            classes.append(c)
            starts.append(i)
            ends.append(i)
            last_class = c

    ends.append(len(result)-1)

    return classes, starts, ends


def compare_segmentation(pred, label, th):
    """
    Temporally compare the predicted and ground-truth segmentations
    """

    pc, ps, pe = get_class_start_end_times(pred)
    lc, ls, le = get_class_start_end_times(label)

    tp = 0
    fp = 0
    matched = [0]*len(lc)
    for i in range(len(pc)):
        inter = np.minimum(pe[i], le) - np.maximum(ps[i], ls)
        union = np.maximum(pe[i], le) - np.minimum(ps[i], ls)
        IoU = (inter/union) * [pc[i] == lc[j] for j in range(len(lc))]

        best_idx = np.array(IoU).argmax()
        if IoU[best_idx] >= th and not matched[best_idx]:
            tp += 1
            matched[best_idx] = 1
        else:
            fp += 1

    fn = len(lc) - sum(matched)

    return tp, fp, fn


def get_eval_score(cfg, preds):
    """
    Compute the evaluation score
    """

    # Path to the annotations
    path_annts = os.path.join(cfg['root_data'], 'annotations')

    eval_type = cfg['eval_type']
    str_score = ""
    if eval_type == 'AVA_ASD':
        groundtruth = os.path.join(path_annts, 'ava_activespeaker_val_v1.0.csv')
        score = run_evaluation_asd(preds, groundtruth)
        str_score = f'{score*100:.2f}%'
    elif eval_type == 'AVA_AL':
        groundtruth = os.path.join(path_annts, 'ava_val_v2.2.csv')
        labelmap = os.path.join(path_annts, 'ava_action_list_v2.2_for_activitynet_2019.pbtxt')
        score = run_evaluation_al(preds, groundtruth, labelmap)
        str_score = f'{score*100:.2f}%'
    elif eval_type == 'AS':
        total = 0
        correct = 0
        threshold = [0.1, 0.25, 0.5]
        tp, fp, fn = [0]*len(threshold), [0]*len(threshold), [0]*len(threshold)

        for video_id, pred in preds:
            # Get a list of ground-truth action labels
            with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f:
                label = [line.strip() for line in f]

            total += len(label)
            for i, lb in enumerate(label):
                if pred[i] == lb:
                    correct += 1

            for i, th in enumerate(threshold):
                tp_, fp_, fn_ = compare_segmentation(pred, label, th)
                tp[i] += tp_
                fp[i] += fp_
                fn[i] += fn_

        acc = correct/total
        str_score = f'(Acc) {acc*100:.2f}%'
        for i, th in enumerate(threshold):
            pre = tp[i] / (tp[i]+fp[i])
            rec = tp[i] / (tp[i]+fn[i])
            f1 = np.nan_to_num(2*pre*rec / (pre+rec))
            str_score += f', (F1@{th}) {f1*100:.2f}%'
    elif eval_type == "VS_max" or eval_type == "VS_avg":

        path_dataset = os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/eccv16_dataset_{cfg["dataset"].lower()}_google_pool5.h5')
        with h5py.File(path_dataset, 'r') as hdf:

            all_f1_scores = []
            all_taus = []
            all_rhos = []
            for video, scores in preds:

                n_samples = hdf.get(video + '/n_steps')[()]
                n_frames = hdf.get(video + '/n_frames')[()]
                gt_segments = np.array(hdf.get(video + '/change_points'))
                gt_samples = np.array(hdf.get(video + '/picks'))
                gt_scores = np.array(hdf.get(video + '/gtscore'))
                user_summaries = np.array(hdf.get(video + '/user_summary'))

                # Take scores from sampled frames to all frames
                gt_samples = np.append(gt_samples, [n_frames - 1]) # To account for last frames within loop
                frame_scores = np.zeros(n_frames, dtype=np.float32)
                for idx in range(n_samples):
                    frame_scores[gt_samples[idx]:gt_samples[idx + 1]] = scores[idx]

                # Calculate segments' avg score and length
                # (Segment_X = video[frame_A:frame_B])
                n_segments = len(gt_segments)
                s_scores = np.empty(n_segments)
                s_lengths = np.empty(n_segments, dtype=np.int32)
                for idx in range(n_segments):
                    s_lengths[idx] = gt_segments[idx][1] - gt_segments[idx][0] + 1
                    s_scores[idx] = (frame_scores[gt_segments[idx][0]:gt_segments[idx][1]].mean())

                # Select for max importance
                final_len = int(n_frames * 0.15) # 15% of total length
                segments = knapsack.fill_knapsack(final_len, s_scores, s_lengths)

                # Mark frames from selected segments
                sum_segs = np.zeros((len(segments), 2), dtype=int)
                pred_summary = np.zeros(n_frames, dtype=np.int8)
                for i, seg in enumerate(segments):
                    pred_summary[gt_segments[seg][0]:gt_segments[seg][1]] = 1
                    sum_segs[i][0] = gt_segments[seg][0]
                    sum_segs[i][1] = gt_segments[seg][1]

                # Calculate F1-Score per user summary
                user_summary = np.zeros(n_frames, dtype=np.int8)
                n_user_sums = user_summaries.shape[0]
                f1_scores = np.empty(n_user_sums)

                for u_sum_idx in range(n_user_sums):
                    user_summary[:n_frames] = user_summaries[u_sum_idx]

                    # F-1
                    tp = pred_summary & user_summary
                    precision = sum(tp)/sum(pred_summary)
                    recall = sum(tp)/sum(user_summary)

                    if (precision + recall) == 0:
                        f1_scores[u_sum_idx] = 0
                    else:
                        f1_scores[u_sum_idx] = (2 * precision * recall * 100 / (precision + recall))

                # Correlation Metrics
                pred_imp_score = np.array(scores)
                ref_imp_scores = gt_scores
                rho_coeff, _ = stats.spearmanr(pred_imp_score, ref_imp_scores)
                tau_coeff, _ = stats.kendalltau(rankdata(-pred_imp_score), rankdata(-ref_imp_scores))

                all_taus.append(tau_coeff)
                all_rhos.append(rho_coeff)

                # Calculate one F1-Score from all user summaries
                if eval_type == "VS_max":
                    f1 = max(f1_scores)
                else:
                    f1 = np.mean(f1_scores)

                all_f1_scores.append(f1)

        f1_score = sum(all_f1_scores) / len(all_f1_scores)
        tau = sum(all_taus) / len(all_taus)
        rho = sum(all_rhos) / len(all_rhos)

        str_score = f"F1-Score = {f1_score}, Tau = {tau}, Rho = {rho}"
    return str_score
