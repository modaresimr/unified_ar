import yaml
import os
from . import printc


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)


Loader.add_constructor('!include', Loader.include)


def load_config(config_filepath):
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        printc(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)


def save_config(config, config_filepath):
    try:
        with open(config_filepath, 'w') as file:
            yaml.dump(config, file)
    except FileNotFoundError:
        printc(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)
