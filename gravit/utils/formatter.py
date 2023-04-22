import os
import glob
import torch
import pickle  #nosec


def get_formatting_data_dict(root_data, graph_name, sp='val'):
    """
    Get a dictionary that is used to format the results following the formatting rules of the evaluation tool
    """

    # Get a list of the feature files
    features = '_'.join(graph_name.split('_')[:-3])
    list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/{sp}/*.pkl')))

    data_dict = {}
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

    return data_dict


def get_formatted_preds(eval_type, data_dict, logits, global_ids):
    """
    Get a list of formatted predictions from the model output, which is used to compute the evaluation score
    """

    # Compute scores from the logits
    scores_all = torch.sigmoid(logits.detach().cpu()).numpy()

    # Iterate over all the nodes and get the formatted predictions for evaluation
    preds = []
    for scores, global_id in zip(scores_all, global_ids):
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

    return preds
