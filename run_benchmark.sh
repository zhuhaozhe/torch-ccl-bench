MODEL_SCRIPT=./all2all_benchmark.py
NUM_CCL_WORKER=${NUM_CCL_WORKER:-1}
HOSTFILE=${HOSTFILE:-hostfile1}
NODE=${NODE:-1}

oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
# export FI_PROVIDER_PATH=/home/haozhe/frameworks.ai.models.intel-models/quickstart/recommendation/pytorch/dlrm/training/cpu/frameworks.ai.pytorch.torch-ccl/oneccl_bindings_for_pytorch/lib/prov

# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export CCL_LOG_LEVEL=info
# export CCL_MNIC=local

export CCL_ATL_TRANSPORT=mpi
export CCL_SCHED_PROFILE=1

# export FI_SHM_DISABLE_CMA=1
# export FI_SHM_USE_DSA_SAR=1
export CCL_ATL_SHM=1

# export CCL_ATL_TRANSPORT=ofi
# export FI_PROVIDER=psm3
# export PSM3_IDENTIFY=1
# export PSM3_ALLOW_ROUTERS=1
# export PSM3_RDMA=1
# export PSM3_PRINT_STATS=1
# export PSM3_RV_MR_CACHE_SIZE=8192
# export PSM3_KASSIST_MODE=none

# export FI_PROVIDER_PATH=/usr/lib64/libfabric
# export CCL_MNIC_NAME=irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2
# export CCL_MNIC_COUNT=2

python -m intel_extension_for_pytorch.cpu.launch $EXTRA_LAUNCH_ARG --enable_tcmalloc --ccl_worker_count $NUM_CCL_WORKER --distributed --hostfile $HOSTFILE --nnodes $NODE \
$MODEL_SCRIPT 2>&1 > verbose.log
wait
python parse_result.py 

# --nprocs-per-node=2 --nodes-list=0 
#  --logical_core_for_ccl