import numpy as np
import xarray as xr
import os

# ----------------------------------
# preprocess - static features
# ----------------------------------

def _norm(x, max_v, min_v):
    """ we assume max_v > 0 & max_v > min_v """
    return (x - min_v) / (max_v - min_v)


def normalize_latlon(latlons):
    norm_latlon = {'lat': {'max_v': 86, 'min_v': 23}, # it does not start from the equator & does not reah the pole
                   'lon': {'max_v': 76, 'min_v': -76}} # it does not consider full earth
    
    latlons[0] = _norm(latlons[0], **norm_latlon['lat'])
    latlons[1] = _norm(latlons[1], **norm_latlon['lon'])

    return latlons


def crop_Dataset(product, x_start, y_start, size=256):
    """ crop a squared region size
        provide upper-left corner with (x_start, y_start)
    """
    return product.isel(nx=slice(x_start, x_start+size), 
                        ny=slice(y_start, y_start+size))


def mk_crop_np(product, x_start, y_start, size=256):
    """ crop a squared region size^2
        provide upper-left corner with (x_start, y_start)
    """
    return product[y_start: y_start + size, x_start: x_start + size]


# ----------------------------------
# load extra information - static features
# ----------------------------------
def get_altitudes(data_path, crop=None, norm=True):
    """ 
    Params:
        crop (dict, optional): {x_start, y_start, size}
    """ 

    altitudes = np.fromfile(data_path, dtype=np.float32)
    altitudes = altitudes.reshape(1019, 2200)
    max_alt = altitudes.max()
    
    if crop is not None:
        altitudes = mk_crop_np(altitudes, **crop)
        
    if norm:
        altitudes[altitudes < 0] = 0  # make under see level 0
        altitudes = altitudes / max_alt  # normalize
    
    return np.expand_dims(altitudes, axis=0), ['altitudes']


def get_lat_lon(data_path, crop=None, norm=True):

    latlons = xr.open_dataset(data_path)
    
    if crop is not None:
        latlons = crop_Dataset(latlons, **crop)
    
    # get only the values form the netcdf4 file
    latlons = [latlons['latitude'][0].values,
               latlons['longitude'][0].values]
    
    if norm:
        latlons = normalize_latlon(latlons)
    
    return np.stack(latlons), ['latitude', 'longitude']


def load_static(attributes, data_paths, crop=None, norm=True):
    """ 
    Params:
        attributes (list): ['l', 'a']
        data_paths (dict): {'l': path, 'a': path}
        crop (dict): {x_start, y_start, size}
        norm (boolean): 
    Returns:
        statics (np.array): [num_attributes, nx, ny]
        descriptions (list): ['latitude', 'longitude', 'altitudes']
    """ 
    
    statics, descriptions = [], []
    func = {'l': get_lat_lon, 'a': get_altitudes}
    
    for attr in attributes:
        data, channels = func[attr](data_paths[attr], crop=crop, norm=norm)
        statics.append(data)
        descriptions += channels
        
    if len(statics) != 0:
        statics = np.concatenate(statics)
    
    return statics, descriptions


if __name__ == "__main__":

    data_path = '/home/yaoyi/lin00786/weather4cast/data/core-w4c/R1/data'
    static_data_path = '/home/yaoyi/lin00786/weather4cast/data/static/'
    splits_path = '/home/yaoyi/lin00786/weather4cast/'

    params = get_params(data_path, static_data_path, splits_path, region_id='R1')
    
    statics, descriptions = load_static(attributes=params['use_static'].split('-'), 
                                    data_paths=params['static_paths'], 
                                    crop=params['crop_static'])
    print(statics.shape)
    print(descriptions)

    output_path = '/home/yaoyi/lin00786/weather4cast/preprocess-data/static'
    output_file = os.path.join(output_path, 'static_{}.h5'.format(params['region_id']))

    with h5py.File(output_file, 'w') as hf:
        dset = hf.create_dataset('data', data=statics)
        dset.attrs['descriptions'] = descriptions
