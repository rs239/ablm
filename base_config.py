#!/usr/bin/env python
import pathlib


__SCRIPT_DIR = pathlib.Path(__file__).parent

##### if you download the repo and put the data in the data/ subfolder, nothing should need to change.
## Otherwise, change these paths as per your system
######
## ABLM_DIR = pathlib.Path('your/ablm/toplevel/dir')
## ABLM_DATADIR = pathlib.Path('your/data/for/ablm')
######

ABLM_DIR = __SCRIPT_DIR
ABLM_DATADIR = ABLM_DIR / 'data'

device = 'cuda:1'
