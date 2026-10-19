# Usage: bash run_split_warm.sh N   -- N-way concurrent build-only with the caches KEPT (warm).
# Run after a previous cold run so ~/.tensile and ~/.tensilelite are populated.
N=${1:-25}
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
cd $TL || exit 1
export OMP_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
python timing_runs/cms/split_experiment.py $N _warm > timing_runs/cms/split_${N}_warm_result.txt 2>&1
echo finished >> timing_runs/cms/split_${N}_warm_result.txt
