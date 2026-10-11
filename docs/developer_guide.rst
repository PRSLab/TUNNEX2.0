Developer Guide
===============

This page details how to contribute to TUNNEX2.0.

Setting up the environment
--------------------------

Create the test environment from the YAML file in ``devtools/conda-envs``:

.. code-block:: bash

   python devtools/scripts/create_conda_env.py -n test -p 3.13 devtools/conda-envs/test_env.yml
   conda activate test

Running the tests
-----------------

.. code-block:: bash

   pytest --cov=tunnex_2

Building the documentation
--------------------------

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

The documentation requires ``sphinx``, ``pydata-sphinx-theme``,
``sphinx-design`` and ``sphinx-copybutton``.