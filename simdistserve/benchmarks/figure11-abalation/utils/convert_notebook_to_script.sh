# this file direction
file_dir=$(dirname $0)
cd $file_dir/../ \
&& jupyter nbconvert --to script 02-draw-rate-abalation.ipynb --output 02-draw_rate_abalation \
&& jupyter nbconvert --to script 03-draw-slo-abalation.ipynb --output 03-draw_slo_abalation