﻿### Outil d'évaluation robco MCN

## Pré-requis

- Python 3.11+

## Usage
```bash
# create venv
python venv venv
# install requirements (linux)
source venv/bin/activate
# (windows)
activate
# run tests
main.py [-h] [--sample_size SAMPLE_SIZE] [--summary_eval_path PATH] [--parallelization_factor PARALLELIZATION_FACTOR] [--ws_throttle WS_THROTTLE] --ws_address WS_ADDRESS
               [--eval_results_folder EVAL_RESULTS_FOLDER]

Evaluation script

options:
  -h, --help            show this help message and exit
  --sample_size SAMPLE_SIZE
                        Sample size per intent dataset
  --summary_eval_path PATH
                        Path to save the eval summary results (json)
  --parallelization_factor PARALLELIZATION_FACTOR
                        Parallelization factor (CPU only)
  --ws_throttle WS_THROTTLE
                        WebSocket throttle
  --ws_address WS_ADDRESS
                        WebSocket address
  --eval_results_folder EVAL_RESULTS_FOLDER
                        The folder to put the eval results in (model input + output)

```