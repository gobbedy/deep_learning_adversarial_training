#!/bin/bash

set -o pipefail
me=$(basename ${0%%@@*})
full_me=${0%%@@*}
me_dir=$(dirname $(readlink -f ${0%%@@*}))

state_dim=$1
architecture=$2

if [[ -z ${state_dim} || -z ${architecture} ]]; then
  echo "Forgot state dim, architecture arguments"
  exit 1
fi

# create output directory tree
datetime_suffix=$(date +%b%d_%H%M%S)
output_dir=${me_dir}/regress/${architecture}${state_dim}_${datetime_suffix}
mkdir ${output_dir}

# simulation parameters
nodes=1
cpus=4 # IGNORED -- automatically get 5 cpus per 2 GPUs
gpus=1 # gpus or gpus per node? not sure
time="00:05:00" # hours:minutes:seconds
mem=20gb # total memory on all processes -- NOTE FOR NOW THIS IS IGNORED
email=yes
job_name="assignment3_simulation"
mbatch_script="${me_dir}/simulation_helios.mbatch"
python_script="${me_dir}/keras_simulation.py"
python_options="-s ${state_dim} -a ${architecture} -o ${output_dir}" # eg -h|--short, -s|--sanity, -p|--profile





# launch job
prologue_file=${output_dir}/${job_name}.prologue
logfile=${output_dir}/${job_name}.log
export="output_dir=\"${output_dir}\",python_options=\""${python_options}"\",python_script=\""${python_script}"\""
mail=''
if [[ ${email} == yes ]]; then
  mail="-M ${EMAIL} -m ba"
fi
test=''
if [[ ${test_mode} == yes ]]; then
  test="--test"
fi

#echo "$me: launching the following command:"
msub -l walltime="${time}" -l nodes=${nodes}:gpus=${gpus} -v "${export}" -N "${job_name}" ${mail} -e "${logfile}" -o "${logfile}" "${mbatch_script}" |& tee -a ${prologue_file}

# :pmem=${mem}