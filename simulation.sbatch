#!/bin/bash

set -o pipefail
me=$(basename ${0%%@@*})
full_me=${0%%@@*}
me_dir=$(dirname $(readlink -f ${0%%@@*}))

######################################################################
# Convenience functions
######################################################################
### exit elegantly
function die {
  err_msg="$@"
  printf "$me: %b\n" "${err_msg}" >&2
  exit 1 
}

# Usage: info "string message"
function info
{
  printf "${me}: INFO - %s\n" "$@"
}

######################################################################
# Get node prefix, node suffix and login node
######################################################################

# get node prefix (eg gra, cdr, nia)
node_prefix=$(echo ${SLURM_JOB_NODELIST} | grep -oP '^[^\[]+')

# get node suffix for python, and login node
# eg in python cedar nodes are "cdr1132.int.cedar.computecanada.ca" instead of "cdr1132"
case "$node_prefix" in
  gra)
    python_node_suffix=''
    login_node="gra-login1"
  ;;
  cdr)
    python_node_suffix='.int.cedar.computecanada.ca'
    login_node="cedar1"
  ;;
  cp)
    python_node_suffix=''
    login_node="ip15"
  ;;
  nia)
    python_node_suffix='.scinet.local'
    login_node="nia-login01"
  ;;
esac

######################################################################
# Setup cleanup routine to trap on any exit condition (even normal)
######################################################################

function cleanupOnExit
{

  error=0
  local signal="$1"
  if [[ "${signal}" != "EXIT" ]]; then
    error=1
    printf "\nERROR in ${me}: User interrupted (${signal})\n" >&2
  fi

  # kill tmux sessions
  if [[ -n ${launched_dispynodes} ]]; then
    for node_name in ${node_name_list[@]} ; do
      ssh ${node_name} "tmux kill-session -t dispynode_session"
    done
  fi

  # summary results for end user
  if [[ -n ${mail} ]]; then
    # send e-mail to end user: since compute node doesn't have internet, must ssh to login node
    ssh ${login_node} "summarize_regression.sh ${error} ${SLURM_JOB_NAME} ${SLURM_JOB_ID} ${output_dir}"
  fi

  info "Done at $(date +%Y-%m-%d.%H:%M:%S)"

  # Cleanup trap signals and force kill on same signal to propagate
  trap - ${signal}
  trap - EXIT
  if [[ "${signal}" != "SIGUSR1" ]]; then
    kill -${signal} $$
  fi
}

trap 'cleanupOnExit "SIGHUP"' SIGHUP
trap 'cleanupOnExit "SIGINT"' SIGINT
trap 'cleanupOnExit "SIGTERM"' SIGTERM
trap 'cleanupOnExit "SIGQUIT"' SIGQUIT
trap 'cleanupOnExit "SIGUSR1"' SIGUSR1
trap 'cleanupOnExit "SIGUSR1"' USR1
trap 'cleanupOnExit "EXIT"' EXIT


######################################################################
# Get list of node names (regular for ssh + pythonic for dispy)
######################################################################

# note: this line needs to be outside if/else statement. see this page for why:
# https://unix.stackexchange.com/questions/45957/what-are-the-scope-restrictions-for-setting-shopt-extglob-and-other-options
shopt -s extglob # allow +() in regular expression

if [[ ${SLURM_JOB_NUM_NODES} -eq 1 ]]; then
  node_name_list=(${SLURM_JOB_NODELIST})
  python_node_name_list=(${SLURM_JOB_NODELIST}${python_node_suffix})
else
  # get node id ranges in comma separated string format
  nodeid_ranges_string=$(echo ${SLURM_JOB_NODELIST} | grep -oP '(?<=\[).*?(?=\])')

  # convert node id ranges string to array of ranges
  IFS=',' read -r -a nodeid_ranges_array <<< "$nodeid_ranges_string"

  # convert array of ranges to array of node names
  for nodeid_range in ${nodeid_ranges_array[@]} ; do
    case "$nodeid_range" in
      +([[:digit:]]) )
        node_name_list+=( ${node_prefix}${nodeid_range} )
        python_node_name_list+=( ${node_prefix}${nodeid_range}${python_node_suffix} )
      ;;
      +([[:digit:]])-+([[:digit:]]) )
        eval node_name_list+=( ${node_prefix}{${nodeid_range/-/..}} )
        eval python_node_name_list+=( ${node_prefix}{${nodeid_range/-/..}}${python_node_suffix} )
      ;;
    esac
  done
fi

# check that length of node name array is equal to the number of nodes
if [[ ${#node_name_list[@]} -ne ${SLURM_JOB_NUM_NODES} ]]; then
  die "List of node names has different size ("${#node_name_list[@]}") than number of nodes ("${SLURM_JOB_NUM_NODES}")"
fi

# check that length of pythonic node name array is equal to the number of nodes
if [[ ${#python_node_name_list[@]} -ne ${SLURM_JOB_NUM_NODES} ]]; then
  die "List of node names has different size ("${#python_node_name_list[@]}") than number of nodes ("${SLURM_JOB_NUM_NODES}")"
fi

######################################################################
# Launch dispynode on each node
######################################################################

for node_name in ${node_name_list[@]} ; do
  dispynode_logfile=${output_dir}/dispy/${node_name}_dispynode.log

  # -f option to make it non-blocking, important as this may otherwise make thousands of cores sit idly
  # as we wait for each ssh command to finish sequentially
  ssh -f ${node_name} "launch_dispynode.sh ${dispynode_logfile}"
done
launch_remote_dispynodes.sh ${output_dir} "${node_name_list[@]}"
launched_dispynodes=1

######################################################################
# Launch pythons simulation
######################################################################
export PYTHONUNBUFFERED=1
#${python_script_path} ${python_options} --compute_nodes_pythonic "${python_node_name_list[@]}"
${python_script_path} ${python_options}