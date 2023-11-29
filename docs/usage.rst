.. _usage:

-----
Usage
-----

``eigenstrapping`` aims to provide the user with a robust generative null model
for null hypothesis testing of spatial brain maps.

These tutorials step through the basics of downloading pre-generated eigenmodes
and loading brain maps (or as ``neuromaps`` calls them - annotations), generating
your first surrogates, and plotting the variogram fits (to check the smoothness).
These tutorials also show you how to generate eigenmodes on your own surfaces
and surrogates using that surface, say if you have a pathological cohort that you
expect has a different geometry to a healthy cohort. If you still have questions after
going through this guide you can refer to the :ref:`api` or ask a question
on `GitHub <https://github.com/SNG-newy/eigenstrapping/issues>`_.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2

   tutorials/tutorial_cortex.rst
   tutorials/tutorial_subcortex.rst
   tutorials/tutorial_ownsurface.rst