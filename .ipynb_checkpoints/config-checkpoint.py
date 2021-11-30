import numpy as np
import os


def prepare_crop(regions, region_id):
    """ this function prepares the expected parameters to crop images per region
        e.g., to crop latitudes to the region of interest 
    """
    
    x, y = regions[region_id]['up_left']
    crop = {'x_start': x, 'y_start': y, 'size': regions[region_id]['size']}
    return crop


def n_static_vars(str_vars):
    """ computes how many static variables will be used
        maximum is 3, lon, lat, alt
    """
    
    if str_vars=='':
        n_static = 0
    else:
        n_static = len(str_vars.split('-'))
        if 'l' in str_vars: 
            n_static += 1  # 'l' loads both lat/lon, so 2 vars (not 1)
    return n_static


def get_params(data_path,
               static_data_path,
               splits_path,
               region_id='R1',
               seq_len=4,
               horizon=32,
               size=256,
               collapse_time=False):

    """ Set paths & parameters to load/transform/save data and models.

    Params:
        data_path (str): path to the parent folder containing folders 
        static_data_path (str): path to the folder containing the static channels
        splits_path (str): path to the folder containing the csv and json files defining the data splits        
        region_id (str, optional): region to load data from. Default: 'R1'.
        seq_len (int, optional): input sequence length. Default: 4.
        horizon (int, optional): output sequence length. Default: 32.
        size (int, optional): size of the region. Default: 256.
        collapse_time (bool, optional): merging 

    Returns:
        dict: contains the params
    """
    
    data_params = {}
    model_params = {}
    train_params = {}
    optimization_params = {}

    regions = {'R3': {'up_left': (935, 400), 'split': 'train', 'desc': 'South West\nEurope', 'size': size}, 
               'R6': {'up_left': (1270, 250), 'split': 'test', 'desc': 'Central\nEurope', 'size': size}, 
               'R2': {'up_left': (1550, 200), 'split': 'train', 'desc': 'Eastern\nEurope', 'size': size},  
               'R1': {'up_left': (1850, 760), 'split': 'train', 'desc': 'Nile Region', 'size': size}, 
               'R5': {'up_left': (1300, 550), 'split': 'test', 'desc': 'South\nMediterranean', 'size': size}, 
               'R4': {'up_left': (1020, 670), 'split': 'test', 'desc': 'Central\nMaghreb', 'size': size},
               'R7': {'up_left': (1700, 470), 'split': 'train', 'desc': 'Bosphorus', 'size': size}, 
               'R8': {'up_left': (750, 670), 'split': 'train', 'desc': 'East\nMaghreb', 'size': size}, 
               'R9': {'up_left': (450, 760), 'split': 'test', 'desc': 'Canarian Islands', 'size': size}, 
               'R10': {'up_left': (250, 500), 'split': 'test', 'desc': 'Azores Islands', 'size': size}, 
               'R11': {'up_left': (1000, 130), 'split': 'test', 'desc': 'North West\nEurope','size': size}
               } 
    print(f'Using data for region {region_id} | size: {size} | {regions[region_id]["desc"]}')

    # ------------
    # 1. Files to load
    # ------------
    if region_id in ['R1', 'R2', 'R3', 'R7', 'R8']:
        track = 'core-w4c'
    else:
        track = 'transfer-learning-w4c'

    data_params['region_id'] = region_id
    data_params['data_path'] = os.path.join(data_path, track, region_id)
    
    data_params['static_paths'] = {}
    data_params['static_paths']['l'] = os.path.join(static_data_path, 
                                                    'Navigation_of_S_NWC_CT_MSG4_Europe-VISIR_20201106T120000Z.nc')
    data_params['static_paths']['a'] = os.path.join(static_data_path, 
                                                    'S_NWC_TOPO_MSG4_+000.0_Europe-VISIR.raw')
    
    data_params['splits_path'] = os.path.join(splits_path, 'splits.csv')
    data_params['test_split_path'] = os.path.join(splits_path, 'test_split.json')
    data_params['blacklist_path'] = os.path.join(splits_path, 'blacklist.json')
    
    # ------------
    # 2. Data params    
    # ------------
    data_params['collapse_time'] = collapse_time
    data_params['use_static'] = 'l-a'  # use '' to not use static features
    data_params['target_vars'] = ['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma']
    data_params['products'] = {'CTTH': ['temperature'], 
                               'CRR': ['crr_intensity'], 
                               'ASII': ['asii_turb_trop_prob'], 
                               'CMA': ['cma']}
    data_params['weights'] = {'temperature': .25, 
                              'crr_intensity': .25, 
                              'asii_turb_trop_prob': .25, 
                              'cma': .25} # to use by the metric
    
    data_params['spatial_dim'] = (size, size)
    data_params['crop_static'] = prepare_crop(regions, region_id)
    data_params['region_id'] = region_id
    data_params['seq_len'] = seq_len  
    data_params['horizon'] = horizon 
    data_params['day_bins'] = 96

    # preprocessing:
    #    a. fill_value: value to replace NaNs (currently temperature is the one that has more)
    #    b. max_value: maximum value of the variable when it's saved on disk as integer
    #    c. scale_factor: netCDF automatically uses this value to re-scale the value
    #    d. add_offset: netCDF automatically uses this value to shift a variable
    #
    # c. and d. together mean that once loaded, the data is in the scale [add_offset, max_value * scale_factor + add_offset]
    # Hence, to normalize the data between [0, 1] we must use:
    #    data = (data - add_offset) / (max_value * scale_factor - add_offset)
    preprocess = {'cma': {'fill_value': 0, 
                          'max_value': 1, 
                          'add_offset': 0, 
                          'scale_factor': 1}, 
                  'temperature': {'fill_value': 0, 
                                  'max_value': 35000, 
                                  'add_offset': 130, 
                                  'scale_factor': np.float32(0.01)}, 
                  'crr_intensity': {'fill_value': 0, 
                                    'max_value': 500, 
                                    'add_offset': 0, 
                                    'scale_factor': np.float32(0.1)},
                  'asii_turb_trop_prob': {'fill_value': 0, 
                                          'max_value': 100, 
                                          'add_offset': 0, 
                                          'scale_factor': 1},
                  'ct': {'categories': {'flag_values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                        'flag_meanings': ['Cloud-free land',
                                                          'Cloud-free sea',
                                                          'Snow over land',
                                                          'Sea ice',
                                                          'Very low clouds', 
                                                          'Low clouds',
                                                          'Mid-level clouds',
                                                          'High opaque clouds',
                                                          'Very high opaque clouds',
                                                          'Fractional clouds',
                                                          'High semitransparent thin clouds',
                                                          'High semitransparent moderately thick clouds',
                                                          'High semitransparent thick clouds',
                                                          'High semitransparent above low or medium clouds',
                                                          'High semitransparent above snow/ice',]}
                        }
                 }
    
    preprocess_tgt = {'cma': {'fill_value': np.nan, 
                              'max_value': 1, 
                              'add_offset': 0, 
                              'scale_factor': 1}, 
                      'temperature': {'fill_value': np.nan, 
                                      'max_value': 35000, 
                                      'add_offset': 130, 
                                      'scale_factor': np.float32(0.01)}, 
                      'crr_intensity': {'fill_value': np.nan, 
                                        'max_value': 500, 
                                        'add_offset': 0, 
                                        'scale_factor': np.float32(0.1)},
                      'asii_turb_trop_prob': {'fill_value': np.nan, 
                                              'max_value': 100, 
                                              'add_offset': 0, 
                                              'scale_factor': 1}}
    
    data_params['preprocess'] = {'source': preprocess, 'target': preprocess_tgt} 
    return data_params


if __name__ == '__main__':
    # this is only executed when the module is run directly.
    print(get_params())
