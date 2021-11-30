import glob
import numpy as np
import numpy.ma as ma
import netCDF4
import pandas as pd
import json
import sys
import os
import time
import h5py

from config import get_params


def mk_crop_dataset_netcdf4(data, x_start, y_start, size=256):
    """ crop a squared region size
        provide upper-left corner with (x_start, y_start)
    """
    return data[y_start: y_start+size, x_start: x_start+size]


def preprocess_numeric(data, fill_value, max_value, add_offset, scale_factor):
    """ scale data into [0, 1]
    
    Returns:
        np.array: [1, height, width]
    """
    data = np.float32(data)
    data = (data - add_offset) / (max_value * scale_factor - add_offset)
    data = data.filled(fill_value)  # fill NaNs with 'fill_value'
    assert 0 <= np.nanmin(data) and np.nanmax(data) <= 1, f"Error, the scale of the variables is wrong"
    return np.expand_dims(data, axis=0)


def to_categorical(data, num_classes):
    """ one-hot encodes a tensor 
        credit: https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
    """
    return np.eye(num_classes, dtype='uint8')[data]


def preprocess_categorical(data, categories):
    """ converting to 1-hot-encoding & move channels at the beginning
    
    Returns:
        np.array: [one-hot channels, height, width]
    """

    # there should not be NaNs in the categorical data
    # if yes, using the undefined class (usually the last class) to fill NaNs
    data = data.filled(0)
    data = to_categorical(data, len(categories))
    data = np.moveaxis(data, -1, 0)  # move the feature channel to the beginning
    return data


# ----------------------------------
# load products or sequences with netCDF4
# ----------------------------------
def get_prod_rename(product):
    """ this is just a terrible hack since folder containing this product 
        and files have a slightly different name 
    """
    if product=='ASII':
        return 'ASII-TF'
    else:
        return product
    

def load_single_netcdf4(product, attributes, 
                        day_in_year, time, data_path, preprocess):
    """ open one *.nc file and return only the specified attributes for one product
    
    Params:
        product (str): name of the product
        attributes (list): contain a list of attributes to retrieve
        day_in_year (int): 
        time (str): 
        data_path (str):
        preprocess (dict): 
        
    Returns:
        ds_vars (dict): {attribute: np.array}
        ds_masks (np.array): [num_attributes, ny, nx]
        ds_attrs (dict): {attribute: [attribute if numeric variable else sub-attributes]}
        if_black (boolean): if the time is in the blacklist (file does not exist) 
    """
    
    ds_vars, ds_attrs, ds_masks, if_black = {}, {}, {}, False

    file = f'S_NWC_{get_prod_rename(product)}_MSG*_Europe-VISIR_*T{time}Z.nc'
    path = f'{data_path}/{day_in_year}/{product}/{file}'
    file_path = glob.glob(path)
    
    if len(file_path) == 0:
        
        if_black = True
        print(f'File does not exist: {path}.')
        
        # generate NaN matrix for the attribute
        for attr in attributes:
            
            if preprocess.get(attr) is not None:
                attr_preproc = preprocess[attr]
                
                if attr_preproc.get('categories') is not None:  # if attr is categorical
                    categories = attr_preproc['categories']['flag_meanings']
                    ds_vars[attr] = np.full((len(categories), 256, 256), 0)
                    ds_masks[attr] = np.full((len(categories), 256, 256), True)
                    ds_attrs[attr] = [attr + '_' + s for s in categories]
                    
                else:  # if attr is numeric
                    ds_vars[attr] = np.full((1, 256, 256), attr_preproc['fill_value'])
                    ds_masks[attr] = np.full((1, 256, 256), True)
                    ds_attrs[attr] = [attr]
            else:
                print(f'Preprocessing for {attr} is missing.')
                raise NotImplementedError        
    else:
        
        ds = netCDF4.Dataset(file_path[0], 'r')
        
        for attr in attributes:

            v = ds.variables[attr][...]
            m = ds.variables[attr][...].mask  # True is NaN
            if not np.any(m):
                m = np.full((1, 256, 256), False)
            else:
                m = np.expand_dims(m, axis=0)

            if preprocess.get(attr) is not None:            
                attr_preproc = preprocess[attr]

                if attr_preproc.get('categories') is not None:  # if attr is categorical
                    categories = attr_preproc['categories']['flag_meanings']
                    ds_vars[attr] = preprocess_categorical(v, categories)
                    ds_masks[attr] = np.repeat(m, len(categories), axis=0)
                    ds_attrs[attr] = [attr + '_' + s for s in categories]
                else:
                    ds_vars[attr] = preprocess_numeric(v, **attr_preproc)
                    ds_masks[attr] = m
                    ds_attrs[attr] = [attr]
            else:
                print(f'Preprocessing for {attr} is missing.')
                raise NotImplementedError
                                
    return ds_vars, ds_masks, ds_attrs, if_black


def load_products_netcdf4(day_in_year, time, products, data_path, preprocess):
    """ loads all products and attributes into a single tensor 
    Params:
        day_in_year (int): the number of day in a year
        time (str): {hour}{minute}00, e.g., '001500'
        products (dict): {product: list(attributes)}
        data_path: 
        crop (dict): contain cropping information, not in use
        preprocess (dict): contain information for preprocessing, 
                           if categorical features, {flag_values, flag_meanings} for one-hot encoding
    Returns:
        prods (np.array): [num_attributes, ny, nx]
        masks (np.array): [num_attributes, ny, nx]
        attrs (list): a list of attributes
        if_blacks (boolean): if the time is in the blacklist (file does not exist) 
    """
    
    prods, masks, attrs, if_blacks = {}, {}, {}, False
    for product, attributes in products.items():
        
        prod, mask, attr, if_black = load_single_netcdf4(product, attributes, 
                                                         day_in_year, time, data_path, preprocess)
        prods = dict(prods, **prod)
        masks = dict(masks, **mask)        
        attrs = dict(attrs, **attr)
        if_blacks |= if_black
    
    prods = np.concatenate(list(prods.values()))
    masks = np.concatenate(list(masks.values()))    
    attrs = np.concatenate(list(attrs.values())).tolist()
    assert prods.shape[0] == masks.shape[0] == len(attrs)
    return prods, masks, attrs, if_blacks  


def get_time():
    """ generate list of time bins """
    return ['{}{}{}{}00'.format('0'*bool(i<10), i, '0'*bool(j<10), j) 
            for i in np.arange(0, 24, 1) for j in np.arange(0, 60, 15)]


def load_sequence_netcdf4(day_in_year, products, data_path, preprocess):
    """ loads the sequence for all products and attributes into a single tensor  
    Params:
        day_in_year (int): the number of day in a year
        products (dict): {product: list(attributes)}
        data_path: 
        preprocess (dict): contain information for preprocessing, 
                           if categorical features, {flag_values, flag_meanings} for one-hot encoding
    Returns:
        sequence (np.ma.array): [len_seq, num_attributes, ny, nx]
        channels (list):
        blacklist (list): a list of time bins that have at least one variable 
    """
        
    prod_seq, mask_seq, blacklist = [], [], []
    bins = get_time()

    for i, time in enumerate(bins):
                 
        prods, masks, channels, if_black = load_products_netcdf4(day_in_year, time, products, 
                                                                 data_path, preprocess)
        prod_seq.append(prods)
        mask_seq.append(masks)        
        
        if if_black:
            blacklist.append(i)
            
    prod_seq = np.stack(prod_seq)
    mask_seq = np.stack(mask_seq)
    sequence = ma.MaskedArray(prod_seq, mask=mask_seq)

    if len(blacklist) > 0:
        print(day_in_year, blacklist)
        
    return sequence, channels, blacklist


if __name__ == "__main__":

    data_path = '/home/yaoyi/lin00786/weather4cast/data/core-w4c/R1/data'
    static_data_path = '/home/yaoyi/lin00786/weather4cast/data/static/'
    splits_path = '/home/yaoyi/lin00786/weather4cast/'

    params = get_params(data_path, static_data_path, splits_path, region_id='R1')

    import argparse
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--year', type=str, default='2019') 
    args = parser.parse_args()

    days = [args.year + '%03d'%i for i in range(366)]
    # days += ['2020'+ '%03d'%i for i in range(366)]
    
    output_path = '/home/yaoyi/lin00786/weather4cast/preprocess-data/R1'

    for day_in_year in days:

        start_time = time.time()

        if not os.path.exists(os.path.join(data_path, day_in_year)):
            continue

        output_file = os.path.join(output_path, f'{day_in_year}.h5')
        
        if os.path.exists(output_file):
            continue
            
        sequence, channels, blacklist = load_sequence_netcdf4(day_in_year=int(day_in_year),
                                                              products=params['products'], 
                                                              data_path=data_path,
                                                              preprocess=params['preprocess']['source'])
        with h5py.File(output_file, 'w') as hf:
            grp = hf.create_group('data')
            grp.create_dataset('value', data=sequence.data)
            grp.create_dataset('mask', data=sequence.mask)
            grp.attrs['descriptions'] = channels
            grp.attrs['blacklist'] = blacklist

        print(f'=== {day_in_year} - {time.time() - start_time} ===')
        
