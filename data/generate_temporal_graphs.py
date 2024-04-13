import os
import glob
import torch
import h5py
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
from random import randint, seed


def get_edge_info(num_frame, args):
    skip = args.skip_factor

    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source = []
    node_target = []
    edge_attr = []
    for i in range(num_frame):
        for j in range(num_frame):
            # Frame difference between the i-th and j-th nodes
            frame_diff = i - j

            # The edge ij connects the i-th node and j-th node
            # Positive edge_attr indicates that the edge ij is backward (negative: forward)
            if abs(frame_diff) <= args.tauf:
                node_source.append(i)
                node_target.append(j)
                edge_attr.append(np.sign(frame_diff))

            # Make additional connections between non-adjacent nodes
            # This can help reduce over-segmentation of predictions in some cases
            elif skip:
                if (frame_diff % skip == 0) and (abs(frame_diff) <= skip * args.tauf):
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

    return node_source, node_target, edge_attr


def generate_sum_temporal_graph(video_data, args, path_graphs):
    """
    Generate temporal graphs of a single video from video summarization data set
    """
    video_id = video_data["global_id"]
    out_folder = video_data["purpose"]
    features = video_data["features"]
    gtscore = np.array([video_data["gtscore"]], dtype=np.float32)[::args.sample_rate]
    gtscore = gtscore.transpose()
    num_samples = features.shape[0]

    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source, node_target, edge_attr = get_edge_info(num_samples, args)

    # x: features
    # g: global_id
    # edge_index: information on how the graph nodes are connected
    # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
    # y: gtscore
    graphs = Data(x=torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32),
                  g=video_id,
                  edge_index=torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                  edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                  y=torch.tensor(gtscore, dtype=torch.float32))

    torch.save(graphs, os.path.join(path_graphs, f'{out_folder}/{video_id}.pt'))


def generate_temporal_graph(data_file, args, path_graphs, actions, train_ids, all_ids):
    """
    Generate temporal graphs of a single video
    """

    video_id = os.path.splitext(os.path.basename(data_file))[0]
    feature = np.transpose(np.load(data_file))
    num_frame = feature.shape[0]

    # Get a list of ground-truth action labels
    with open(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth/{video_id}.txt')) as f:
        label = [actions[line.strip()] for line in f]

    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source, node_target, edge_attr = get_edge_info(num_frame, args)

    # x: features
    # g: global_id
    # edge_index: information on how the graph nodes are connected
    # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
    # y: labels
    graphs = Data(x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
                  g = all_ids.index(video_id),
                  edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                  edge_attr = torch.tensor(edge_attr, dtype=torch.float32),
                  y = torch.tensor(np.array(label, dtype=np.int64)[::args.sample_rate], dtype=torch.long))

    if video_id in train_ids:
        torch.save(graphs, os.path.join(path_graphs, 'train', f'{video_id}.pt'))
    else:
        torch.save(graphs, os.path.join(path_graphs, 'val', f'{video_id}.pt'))


if __name__ == "__main__":
    """
    Generate temporal graphs from the extracted features
    """

    parser = argparse.ArgumentParser()
    # Default paths for the training process
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset', default='50salads')
    parser.add_argument('--features',      type=str,   help='Name of the features', required=True)

    # Hyperparameters for the graph generation
    parser.add_argument('--tauf',          type=int,   help='Maximum frame difference between neighboring nodes', required=True)
    parser.add_argument('--skip_factor',   type=int,   help='Make additional connections between non-adjacent nodes', default=10)
    parser.add_argument('--sample_rate',   type=int,   help='Downsampling rate for the input', default=2)

    args = parser.parse_args()

    print ('This process might take a few minutes')

    actions = {}
    all_ids = []
    if args.dataset == "50salads":

        # Build a mapping from action classes to action ids
        with open(os.path.join(args.root_data, f'annotations/{args.dataset}/mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                actions[cls] = int(aid)

        # Get a list of all video ids
        all_ids = sorted([os.path.splitext(v)[0] for v in
                          os.listdir(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth'))])

        # Iterate over different splits
        list_splits = sorted(os.listdir(os.path.join(args.root_data, f'features/{args.features}')))
        for split in list_splits:
            # Get a list of training video ids
            with open(os.path.join(args.root_data, f'annotations/{args.dataset}/splits/train.{split}.bundle')) as f:
                train_ids = [os.path.splitext(line.strip())[0] for line in f]

            path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.tauf}_{args.skip_factor}/{split}')
            os.makedirs(os.path.join(path_graphs, 'train'), exist_ok=True)
            os.makedirs(os.path.join(path_graphs, 'val'), exist_ok=True)

            list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{split}/*.npy')))

            with Pool(processes=20) as pool:
                pool.map(partial(generate_temporal_graph, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids), list_data_files)

            print (f'Graph generation for {split} is finished')

    elif args.dataset == "SumMe" or args.dataset == "TVSum":

        path_dataset = os.path.join(args.root_data,
                                    f'annotations/{args.dataset}/{args.features}.h5')

        with h5py.File(path_dataset, 'r') as hdf:
            all_videos = list(hdf.keys())

            all_ids = []
            dataset = [None] * len(all_videos)
            for video in all_videos:
                id = int(video.split("_")[1])
                all_ids.append(id - 1)

                data = {}
                data["global_id"] = id
                data["purpose"] = "train"
                data["features"] = np.array(hdf.get(video + '/features'))
                data["gtscore"] = np.array(hdf.get(video + '/gtscore'))
                dataset[id - 1] = data

        # Set the seed value
        seed(42)

        amount_to_select = len(all_ids) // 5
        for split_i in range(1, 6):
            # Init
            for i in all_ids:
                dataset[i]['purpose'] = "train"

            # Randomly select 20% of videos for validation
            train_ids = all_ids.copy()
            for _ in range(amount_to_select):
                train_idx = randint(0, len(train_ids) - 1)
                video_idx = train_ids.pop(train_idx)
                dataset[video_idx]["purpose"] = "val"

            path_graphs = os.path.join(args.root_data,
                                       f'graphs/{args.dataset}_{args.tauf}_{args.skip_factor}/split{split_i}')
            os.makedirs(os.path.join(path_graphs, 'train'), exist_ok=True)
            os.makedirs(os.path.join(path_graphs, 'val'), exist_ok=True)

            with Pool(processes=20) as pool:
                pool.map(partial(generate_sum_temporal_graph, args=args, path_graphs=path_graphs), dataset)

            print(f'Graph generation for split{split_i} is finished')