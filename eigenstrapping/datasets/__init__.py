from .base import (load_surface_examples, load_distmat,
                   load_genepc, fetch_data, load_native_tutorial,
                   load_subcort, txt2memmap, load_memmap)

get_surface_examples = load_surface_examples

__all__ = ['load_surface_examples',
           'get_surface_examples',
           'load_genepc',
           'fetch_data',
           'load_distmat',
           'load_native_tutorial',
           'load_subcort',
           'txt2memmap',
           'load_memmap',
           ]
