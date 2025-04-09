import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--config', default='config/human36m_train.yaml', 
                        type=str, metavar='PATH')

    # General arguments
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-cf','--checkpoint-frequency', default=1, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--visualize', action='store_true')

    # Model arguments
    parser.add_argument('-e', '--epochs', default=400, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=0.00006, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.993, type=float, metavar='LR', help='learning rate decay per epoch')

    parser.add_argument('-num_proposals', type=int, default=1, metavar='N')
    parser.add_argument('-timesteps', type=int, default=1, metavar='N')

    parser.add_argument('-s', '--save', type=str, default="latest_epoch.bin", metavar='FILENAME')
    parser.add_argument('-w', '--webdataset_path', type=str, default="webdataset", metavar='path')
    parser.add_argument('-seed', type=int, default=1234, metavar='N')
    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args