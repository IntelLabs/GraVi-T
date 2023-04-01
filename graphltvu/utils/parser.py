import yaml
import argparse


def get_args():
    """
    Get the command-line arguments for the configuration
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',           type=str,   help='Path to the configuration file', required=True)

    # Additional arguments from the command line override the configuration from cfg file

    # Root directories for the training process
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')

    # Names required for the training process
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment')
    parser.add_argument('--model_name',    type=str,   help='Name of the model')
    parser.add_argument('--graph_name',    type=str,   help='Name of the graphs')
    parser.add_argument('--loss_name',     type=str,   help='Name of the loss function')

    # Other hyper-parameters
    parser.add_argument('--num_modality',  type=int,   help='Number of input modalities')
    parser.add_argument('--channel1',      type=int,   help='Filter dimension of the first GCN layers')
    parser.add_argument('--channel2',      type=int,   help='Filter dimension of the rest GCN layers')
    parser.add_argument('--proj_dim',      type=int,   help='Dimension of the projected spatial feature')
    parser.add_argument('--final_dim',     type=int,   help='Dimension of the final output')
    parser.add_argument('--dropout',       type=float, help='Dropout for the last GCN layers')
    parser.add_argument('--lr',            type=float, help='Initial learning rate')
    parser.add_argument('--wd',            type=float, help='Weight decay value for regularization')
    parser.add_argument('--batch_size',    type=int,   help='Batch size during the training process')
    parser.add_argument('--sch_param',     type=int,   help='Parameter for lr_scheduler')
    parser.add_argument('--num_epoch',     type=int,   help='Total number of epochs')

    return parser.parse_args()


def get_cfg(args):
    """
    Initialize the configuration given the optional command-line arguments
    """

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        delattr(args, 'cfg')

    for k, v in vars(args).items():
        if v is None:
            if k not in cfg:
                raise ValueError(f'Please specify "{k}" in your command-line arguments or in the configuration file')
        else:
            cfg[k] = v

    return cfg
