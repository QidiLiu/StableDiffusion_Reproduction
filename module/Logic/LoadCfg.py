import yaml
from typing import Tuple, Dict

def SafelyReadYaml(in_cfg_path: str) -> Dict:
    with open(in_cfg_path, 'r') as cfg_file:
        try:
            out_cfg = yaml.safe_load(cfg_file)
        except yaml.YAMLError as e:
            print(e)

    return out_cfg


def LoadCfg(in_cfg_path: str, in_mode: str) -> Tuple[Dict, Dict, Dict]:
    '''
    Read configuration files

    Args:
        in_cfg_path: filepath of main configuration file
        in_mode: use configuration file for 'train' mode or 'test' mode
    
    Returns:
        A tuple including dataset configuration, model configuration, and process configuration.
    '''
    main_cfg = SafelyReadYaml(in_cfg_path)
    dataset_cfg_path = main_cfg[in_mode]['dataset_cfg_path']
    model_cfg_path = main_cfg['model_cfg_path']
    proc_cfg_path = main_cfg[in_mode]['proc_cfg_path']

    out_dataset_cfg = SafelyReadYaml(dataset_cfg_path) # Load configuration to prepare dataset
    out_model_cfg = SafelyReadYaml(model_cfg_path) # Load configuration to prepare model
    out_proc_cfg = SafelyReadYaml(proc_cfg_path) # Load configuration to initialize train/test process

    return out_dataset_cfg, out_model_cfg, out_proc_cfg


if __name__ == '__main__':
    dataset_cfg, model_cfg, proc_cfg = LoadCfg('config/main.yaml', 'train')
    #dataset_cfg, model_cfg, proc_cfg = LoadCfg('config/main.yaml', 'test')
    print('--- dataset_cfg ---')
    print(dataset_cfg)
    print('')
    print('--- model_cfg ---')
    print(model_cfg)
    print('')
    print('--- proc_cfg ---')
    print(proc_cfg)
    print('')
