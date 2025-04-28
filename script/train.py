import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from module.Logic.LoadCfg import LoadCfg
from module.TrainerAndTester import *

def main():
    # Load configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/main.yaml', help='Configuration filepath (.yaml)')
    args = parser.parse_args()
    dataset_cfg, model_cfg, train_cfg = LoadCfg(args.cfg, 'train')

    # Initialize and execute training process
    trainer_name = train_cfg['train']['trainer']
    trainer = globals()[trainer_name](dataset_cfg, model_cfg)
    trainer.Train(train_cfg)


if __name__ == '__main__':
    main()
    print('[DEBUG] Wow, it works!')
