import os
import glob
import torch
import pickle  #nosec


def get_formatting_data_dict(cfg):
    """
    Get a dictionary that is used to format the results following the formatting rules of the evaluation tool
    """

    root_data = cfg['root_data']
    dataset = cfg['dataset']
    data_dict = {}

    if 'AVA' in cfg['eval_type']:
        # Get a list of the feature files
        features = '_'.join(cfg['graph_name'].split('_')[:-3])
        list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/val/*.pkl')))

        for data_file in list_data_files:
            video_id = os.path.splitext(os.path.basename(data_file))[0]

            with open(data_file, 'rb') as f:
                data = pickle.load(f) #nosec

            # Get a list of frame_timestamps
            list_fts = sorted([float(frame_timestamp) for frame_timestamp in data.keys()])

            # Iterate over all the frame_timestamps and retrieve the required data for evaluation
            for fts in list_fts:
                frame_timestamp = f'{fts:g}'
                for entity in data[frame_timestamp]:
                    data_dict[entity['global_id']] = {'video_id': video_id,
                                                      'frame_timestamp': frame_timestamp,
                                                      'person_box': entity['person_box'],
                                                      'person_id': entity['person_id']}
    elif 'AS' in cfg['eval_type']:
        # Build a mapping from action ids to action classes
        data_dict['actions'] = {}
        with open(os.path.join(root_data, 'annotations', dataset, 'mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                data_dict['actions'][int(aid)] = cls

        # Get a list of all video ids
        data_dict['all_ids'] = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(root_data, f'annotations/{dataset}/groundTruth'))])

    return data_dict


def get_formatted_preds(cfg, logits, g, data_dict):
    """
    Get a list of formatted predictions from the model output, which is used to compute the evaluation score
    """

    eval_type = cfg['eval_type']
    preds = []
    if 'AVA' in eval_type:
        # Compute scores from the logits
        scores_all = torch.sigmoid(logits.detach().cpu()).numpy()

        # Iterate over all the nodes and get the formatted predictions for evaluation
        for scores, global_id in zip(scores_all, g):
            data = data_dict[global_id]
            video_id = data['video_id']
            frame_timestamp = float(data['frame_timestamp'])
            x1, y1, x2, y2 = [float(c) for c in data['person_box'].split(',')]

            if eval_type == 'AVA_ASD':
                # Line formatted following Challenge #2: http://activity-net.org/challenges/2019/tasks/guest_ava.html
                person_id = data['person_id']
                score = scores.item()
                pred = [video_id, frame_timestamp, x1, y1, x2, y2, 'SPEAKING_AUDIBLE', person_id, score]
                preds.append(pred)

            elif eval_type == 'AVA_AL':
                # Line formatted following Challenge #1: http://activity-net.org/challenges/2019/tasks/guest_ava.html
                for action_id, score in enumerate(scores, 1):
                    pred = [video_id, frame_timestamp, x1, y1, x2, y2, action_id, score]
                    preds.append(pred)
    elif 'AS' in eval_type:
        tmp = logits
        if cfg['use_ref']:
            tmp = logits[-1]

        tmp = torch.softmax(tmp.detach().cpu(), dim=1).max(dim=1)[1].tolist()

        # Upsample the predictions to fairly compare with the ground-truth labels
        preds = []
        for pred in tmp:
            preds.extend([data_dict['actions'][pred]] * cfg['sample_rate'])

        # Pair the final predictions with the video_id
        (g,) = g
        video_id = data_dict['all_ids'][g]
        preds = [(video_id, preds)]

    elif 'VS' in eval_type:
        tmp = logits
        tmp = torch.sigmoid(tmp.squeeze().cpu()).numpy().tolist()
        (g,) = g
        preds.append([f"video_{g}", tmp])

    return preds
