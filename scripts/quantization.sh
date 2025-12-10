export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python tools/quantization.py --args_path configs/quantization.yml