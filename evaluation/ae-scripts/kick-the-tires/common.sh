set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate distserve

export USE_DUMMY_WEIGHT=0
