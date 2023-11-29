#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for downloading datasets from OSF
"""
import os
import re
from collections import defaultdict
from pkg_resources import resource_filename
import json

MATCH = re.compile(
    r'space-(\S+)_(?:den|res)-(\d+[k|m]{1,2})_'
)

def _groupby_match(fnames):
    """"
    Groups files in `fnames` by (space, res/den)

    Parameters
    ----------
    fnames : list-of-str
        Filenames to be grouped
    return_single : bool, optional
        If there is only group of filenames return a list instead of a dict.
        Default: False

    Returns
    -------
    groups : dict-of-str
        Where keys are tuple (source, desc, space, res/den) and values are
        lists of filenames
    """

    out = defaultdict(list)
    for fn in fnames:
        out[MATCH.search(fn).groups()].append(fn)

    out = {k: v if len(v) > 1 else v[0] for k, v in out.items()}
    
    if len(out) == 1:
        out = list(out.values())[0]

    return out

def _osfify_urls(data):
    """
    Formats `data` object with OSF API URL

    Parameters
    ----------
    data : object
        If dict with a `url` key, will format OSF_API with relevant values
    return_restricted : bool, optional
        Whether to return restricted annotations. These will only be accesible
        with a valid OSF token. Default: True

    Returns
    -------
    data : object
        Input data with all `url` dict keys formatted
    """

    OSF_API = "https://files.osf.io/v1/resources/{}/providers/osfstorage/{}"

    if isinstance(data, str) or data is None:
        return data
    elif 'url' in data:
        # if url is None then we this is a malformed entry and we should ignore
        if data['url'] is None:
            return
        # if the url isn't a string assume we're supposed to format it
        elif not isinstance(data['url'], str):
            data['url'] = OSF_API.format(*data['url'])

    try:
        for key, value in data.items():
            data[key] = _osfify_urls(value)
    except AttributeError:
        for n, value in enumerate(data):
            data[n] = _osfify_urls(value)
        # drop the invalid entries
        data = [d for d in data if d is not None]

    return data

def get_data_dir(data_dir=None):
    """
    Gets path to eigenstrapping directory

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'EIGEN_DATA'; if that is not set, will
        use `~/neuromaps-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory
    """

    if data_dir is None:
        data_dir = os.environ.get('EIGEN_DATA',
                                  os.path.join('~', 'eigenstrapping-data'))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir

def _match_files(info, **kwargs):
    """
    Matches datasets in `info` to relevant keys

    Parameters
    ----------
    info : list-of-dict
        Information on files
    kwargs : key-value pairs
        Values of data in `info` on which to match

    Returns
    -------
    matched : list-of-dict
        Annotations with specified values for keys
    """

    # tags should always be a list
    tags = kwargs.get('tags')
    if tags is not None and isinstance(tags, str):
        kwargs['tags'] = [tags]
        
    # 'den' and 'res' are a special case because these are mutually exclusive
    # values (only one will ever be set for a given annotation) so we want to
    # match on _either_, not both, if and only if both are provided as keys.
    # if only one is specified as a key then we should exclude the other!
    denres = []
    for vals in (kwargs.get('den'), kwargs.get('res')):
        vals = [vals] if isinstance(vals, str) else vals
        if vals is not None:
            denres.extend(vals)

    out = []
    for dset in info:
        match = True
        for key in ('space', 'hemi', 'tags', 'format'):
            comp, value = dset.get(key), kwargs.get(key)
            if value is None:
                continue
            elif value is not None and comp is None:
                match = False
            elif isinstance(value, str):
                if value != 'all':
                    match = match and comp == value
            else:
                func = all if key == 'tags' else any
                match = match and func(f in comp for f in value)
        if len(denres) > 0:
            match = match and (dset.get('den') or dset.get('res')) in denres
        if match:
            out.append(dset)

    return out

def get_dataset_info(name):
    """
    Returns information for requested dataset `name`

    Parameters
    ----------
    name : str
        Name of dataset

    Returns
    -------
    dataset : dict or list-of-dict
        Information on requested data
    """

    fn = resource_filename('eigenstrapping',
                           os.path.join('datasets', 'osf.json'))
    with open(fn) as src:
        osf_resources = _osfify_urls(json.load(src))

    try:
        resource = osf_resources[name]
    except KeyError:
        raise KeyError("Provided dataset '{}' is not valid. Must be one of: {}"
                       .format(name, sorted(osf_resources.keys())))

    return resource