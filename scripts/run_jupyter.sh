#!/bin/bash
echo `which python`
python -m jupyter nbextension enable --py --sys-prefix widgetsnbextension

python -m jupyter lab --no-browser --ip=0.0.0.0 --port=1370 --NotebookApp.token='.g^VO0>rq6a}XM&v`479LX*G<9R:dR_pYo+0Y)%.F1dh49B4f1_|K;SbF/&Sp+' & disown
