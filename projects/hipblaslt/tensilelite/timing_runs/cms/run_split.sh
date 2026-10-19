# Usage: bash run_split.sh N   -- cold caches, then the N-way concurrent build-only experiment.
N=${1:-5}
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
cd $TL || exit 1
export OMP_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite /tmp/tl_split_out_${N}_*
python timing_runs/cms/split_experiment.py $N > timing_runs/cms/split_${N}_result.txt 2>&1
echo finished >> timing_runs/cms/split_${N}_result.txt
