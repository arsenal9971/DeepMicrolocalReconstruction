"""
The :mod:`shared` module contains functionality used in all models
"""

# Author: Ingo Guehring

import os


def create_increasing_dir(path_to_parentdir, name):
    i = 0
    while True:
        try:
            path_to_dir = path_to_parentdir + "/" + name + "_{}".format(i)
            os.makedirs(path_to_dir)
        except FileExistsError:
            # directory already exists
            i += 1
            pass
        else:
            break

    return path_to_dir
