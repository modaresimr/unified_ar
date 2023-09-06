
import subprocess
import sys
import os
import unittest
from pathlib import Path
import os
import pkg_resources
# _REQUIREMENTS_PATH = Path(__file__).parent.with_name("requirements.txt")

"""Test that each required package is available."""
# Ref: https://stackoverflow.com/a/45474387/
# requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
requirements=open('requirements.txt', 'r').readlines()

def install_libs():
    try:
        pkg_resources.require('tqdm')
    except:
        os.system('pip install --upgrade -q tqdm')

    from tqdm import tqdm
    pbar = tqdm(requirements)
    for pack in pbar:
        try:
            pkg_resources.require(pack)
        except Exception  as ex:
            print(ex)
            pbar.set_description("Installing %s" % pack)
            os.system('pip install --upgrade -q '+pack)
    pbar.set_description("The required libraries have been  installed")
    pbar.update(len(requirements))
    print("=================== The required libraries have been  installed ===================")



def install_libs2():
    
    reqs =subprocess.check_output(['pip', 'install','--upgrade', '-q','tqdm']) #for progressbars
    from tqdm import tqdm
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    packages=[
        'numpy',
        'pandas',
        'wget',
        'ipympl',
        'intervaltree',
        'tensorflow',
        'tensorflow-plot',
        'scikit-optimize',
        'matplotlib',
        'seaborn',
        'plotly',
		'import-ipynb',
        'memory_profiler',
        'ward-metrics'
    ]
    pbar = tqdm(packages)
    for pack in pbar:
        pbar.set_description("Installing %s" % pack)
        packname=pack.split('<')[0]
        if not(pack in installed_packages):
            os.system('pip install --upgrade -q '+pack)
    pbar.set_description("Everything Installed")
    pbar.update(len(packages))

install_libs()



def install_lab_libs():
    os.system('export NODE_OPTIONS=--max-old-space-size=4096')
    os.system('jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build')
    os.system('jupyter labextension install jupyterlab-plotly --no-build')
    os.system('jupyter labextension install plotlywidget --no-build')
    os.system('jupyter labextension install jupyter-matplotlib --no-build')
    

    os.system('jupyter lab build')
    os.system('unset NODE_OPTIONS')
    

# status=subprocess.check_output(['jupyter', 'labextension', 'check', 'plotlywidget'])
# if("enabled" in status):
#     print('Skip! labextensions are installed');
# else:
#     pass#install_lab_libs();

