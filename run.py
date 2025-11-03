import argparse
from MyLogger import DualLogger
from datasets import *
from model import Model
from train import train
from utils import set_random_seed
from configure import set_default_config
import warnings

warnings.simplefilter("ignore")


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = DualLogger(root=f"./logs/{config['dataset']}",
                        filename=config['log_name'],
                        show_time=True)
    # Load data
    data_root = "./datasets"

    mydata = myDataset(data_name=config['dataset'],
                       root=data_root,
                       aligned_ratio=config['aligned_ratio'])
    config['n_samples'] = mydata.n_samples
    config['n_classes'] = mydata.n_classes
    config['n_views'] = mydata.n_views
    config['in_dim'] = mydata.dim_list

    logger.write('Config:')
    for (key, value) in config.items():
        logger.write(f'- {key:>15} : {value}')

    scores = {'acc': [], 'nmi': [], 'ari': [], 'pur': []}

    if config['train']:
        seeds = [5, 15, 25, 35, 45]
    else:
        seeds = [5]
    for seed in seeds:
        set_random_seed(seed)
        logger.write(f"===================== SEED {seed:<2} ====================")

        # Build model
        model = Model(in_dim=config['in_dim'],
                      emb_dim=config['emb_dim'],
                      hidden_dim=config['hidden_dim'],
                      activation=config['activation'],
                      batchnorm=config['batchnorm'],
                      device=device)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=float(config['lr']),
                                     weight_decay=float(config['weight_decay']))
        if seed == seeds[0]:
            logger.write(model)
            logger.write(optimizer)

        data_loader = mydata.load_train_data(batch_size=config['batch_size'])
        result = train(model, optimizer, data_loader, config, logger, seed, device)

        scores['acc'].append(result[0])
        scores['nmi'].append(result[1])
        scores['ari'].append(result[2])
        scores['pur'].append(result[3])
    logger.write(f"Dataset: {config['dataset']}, Aligned_ratio: {config['aligned_ratio']}")
    logger.write(f"- ACC: {np.mean(scores['acc']) * 100:.2f} ± {np.std(scores['acc']) * 100:.2f}")
    logger.write(f"- NMI: {np.mean(scores['nmi']) * 100:.2f} ± {np.std(scores['nmi']) * 100:.2f}")
    logger.write(f"- ARI: {np.mean(scores['ari']) * 100:.2f} ± {np.std(scores['ari']) * 100:.2f}")
    logger.write(f"- PUR: {np.mean(scores['pur']) * 100:.2f} ± {np.std(scores['pur']) * 100:.2f}")
    logger.write(f"=============== SMART Training Over ==============")


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HandWritten', help='name of dataset.')
    parser.add_argument('--datasetid', type=int, default=0, help='Dataset number in datasets dictionary.')
    parser.add_argument('--aligned_ratio', type=float, default=0.5, help='Aligned ratio of dataset.')
    parser.add_argument('--emb_dim', type=int, default=50, help='Dimension of the learned embeddings')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch Size.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay rate.')
    parser.add_argument('--lambda1', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Loss balance parameter.')

    args = parser.parse_args()

    datasets = {
        0: "BDGP", 1: "HandWritten", 2: "MNIST-USPS", 3: "Wiki",
        4: "NUS-WIDE", 5: "Reuters_dim10", 6: "Hdigit", 7: "Deep Animal",
    }

    args.deviceid = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.deviceid)

    args.datasetid = 0
    args.dataset = datasets[args.datasetid]

    config = set_default_config(args.dataset)
    config['train'] = True      # True: 5 seeds; False: 1 seed

    # config['lr'] = args.lr
    # config['aligned_ratio'] = args.aligned_ratio
    # config['batch_size'] = args.batch_size
    # config['lambda1'] = args.lambda1
    # config['lambda2'] = args.lambda2
    # config['emb_dim'] = args.emb_dim

    config['log_name'] = f"{config['dataset']}_test_20250926_log.txt"

    main(config)
