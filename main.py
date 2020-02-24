import os

import luigi
import numpy as np
import gokart

import novelty_enhanced_bpr

if __name__ == '__main__':
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/base.ini')

    model_name = os.environ.get('MODEL_NAME', '')
    if model_name == 'bpr':
        luigi.configuration.LuigiConfigParser.add_config_path('./conf/bpr.ini')
    elif model_name == 'novelty_enhanced_bpr':
        luigi.configuration.LuigiConfigParser.add_config_path('./conf/novelty_enhanced_bpr.ini')
    else:
        raise NotImplementedError
    np.random.seed(57)
    gokart.run()

