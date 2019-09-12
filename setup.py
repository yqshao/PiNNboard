from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools


setuptools.setup(
    name="tensorboard_plugin_pinnboard",
    version="dev",
    description="PiNN plugin for Tensorboard",
    packages=["tensorboard_plugin_pinnboard"],
    package_data={
        "tensorboard_plugin_pinnboard": ["static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "pinnboard = tensorboard_plugin_pinnboard.plugin:PiNNboard",
        ],
    },
)
