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
    dataset_cfg, model_cfg, test_cfg = LoadCfg(args.cfg, 'test')

    # Prepare and execute training process
    tester_name = test_cfg['test_tester']
    tester = globals()[tester_name](dataset_cfg, model_cfg)
    tester.Test(test_cfg)


if __name__ == '__main__':
    main()
