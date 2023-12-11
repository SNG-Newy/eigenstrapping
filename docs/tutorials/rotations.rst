.. _usage_rotations:

Eigenmode rotations
===================

Core to the ``eigenstrapping`` method is the ability to independently rotate
groups of eigenmodes, which is performed through the :func:`eigenstrapping.rotations.rotate_matrix`
function. This function will take a 2D array of arbitrary size and rotate it by
a random angle:

.. code-block:: py
    
    >>> from eigenstrapping import rotations, datasets
    
    >>> 

Saving rotations
----------------

Beyond calculating eigenmodes and eigenvalues, ``eigenstrapping`` can generate
and store random rotations of eigenmodes in order for the user to preserve the
null resampling method - say, if one wants to maintain consistency of null topology
between fMRI volumes.




