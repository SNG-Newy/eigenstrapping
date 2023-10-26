# eigenstrapping
A toolbox for generating spatial null maps using geometric eigenmodes

# Installation
Clone the repository into the directory of your choice and navigate to the repository. 

To generate your first surrogates:
```python
from eigenstrapping import SurfaceEigenstrapping
data = 'datasets/hcp_example_contrast.txt'
surface = 'datasets/surfaces/standard/fsaverage.L.pial.fs_LR_32k.surf.gii'
eigen = SurfaceEigenstrapping(
          data,
          surface,
          num_modes=500,
          resample=True,
          permute=True,
          )
surrs = eigen(1000)
```

Perform a non-parametric p-value calculation on a target map:
```python
import numpy as np
import matplotlib.pyplot as plt
from stats import pearsonr, calc_pval
target = 'datasets/myelin.txt'
stat = pearsonr(data, target)
pval, perms = calc_pval(target, data, surrs)
print(f'p-value: ${pval:0.3f}')
plt.hist(perms, bins=50)
plt.axvline(stat, color='k', linestyle='dashed')
```

You can also construct surrogate data parcellations:
```python
from utils import parcellate
orig_graph = parcellate(data, atlas='schaefer', density='400')
graphs = parcellate(surrs, atlas='schaefer', density='400')
# compute features like modularity or degree etc.
```

