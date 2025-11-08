export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python tools/train.py --args_path configs/train.yml