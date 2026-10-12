Getting Started
===============

Installation
------------

Create a conda environment and install the package in development mode:

.. code-block:: bash

   conda create -n tunnex_2 python=3.13
   conda activate tunnex_2
   pip install -e .

Running TUNNEX2.0
-----------------

TUNNEX2.0 is a command-line program. Run it as a module:

.. code-block:: bash

   python -m tunnex_2 [options] files ...

The positional argument ``files`` lists the input files for the computations.

Command-line options
--------------------

.. code-block:: text

   usage: tunnex_2 [-h] [-o] [-z] [-f] [-y] [-e] [-i] [-q] [-p] [-c] [files ...]

   positional arguments:
     files                 input files for tunnex_2 computations

   options:
     -h, --help            show this help message and exit
     -o, --orca            use ORCA to compute the IRC data (default is Gaussian)
     -z, --no-zpve, --no_zpve
                           do not calculate ZPVE
     -f, --calcfc          faster but less accurate method for projected
                           frequencies (only for Gaussian)
     -y, --hybrid          scale IRC curve using stationary points (cannot be
                           used with Eckart potential)
     -e, --eckart          use Eckart potential to compute the IRC data
     -i, --iso             use computed Hessians to calculate frequencies
                           (isotopic substitution analysis)
     -q, --qmt-only, --qmt_only
                           run the QMT computations for the already computed
                           IRC path
     -p, --no-qmt, --no_qmt
                           compute only IRC data without running the QMT
                           computations
     -c, --comp            use already computed files instead of running
                           calculations

Common workflows
----------------

Full calculation with Gaussian (default):

.. code-block:: bash

   python -m tunnex_2 input_ts_guess

Full calculation with ORCA:

.. code-block:: bash

   python -m tunnex_2 -o input_ts_guess

Compute only the IRC data, without the QMT step:

.. code-block:: bash

   python -m tunnex_2 -p input_ts_guess

Compute the IRC data, using Eckart potential:

.. code-block:: bash

   python -m tunnex_2 -p input_ts_guess input_reactant_guess input_product_guess

Compute the IRC data, using Eckart potential (with already computed files):

.. code-block:: bash

   python -m tunnex_2 -p input_ts_guess computed_ts computed_reactant computed_product

Reuse already computed files instead of running the calculations again:

.. code-block:: bash

   python -m tunnex_2 -c input_file computed_ts
   python -m tunnex_2 -c input_file computed_ts computed_irc
   python -m tunnex_2 -c input_file computed_ts computed_irc computed_reactant computed_product

Run only the QMT step for an IRC path that has already been computed:

.. code-block:: bash

   python -m tunnex_2 -q tunnex_input_file

Notes
-----

* ``--no_zpve``, ``--calcfc``, and ``--hybrid`` cannot be combined with ``--eckart``.
* ``--calcfc`` applies only to Gaussian calculations.
* ``--qmt-only`` and ``--no-qmt`` select opposite parts of the workflow
  and are not meant to be used together. In fact, ``--qmt-only`` cannot
  be combined with any other flag.

Input files
-----------

The required input files and their format will be described later.

Output
------

The output files and their format will be described later.