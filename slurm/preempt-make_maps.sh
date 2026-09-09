#!/bin/bash
preempt_handler()
{
    #place here: commands to run when preempt signal (SIGTERM) arrives from slurm
    kill -TERM ${1} #forward SIGTERM signal to the user application
    #if --requeue was used, slurm will automatically do so here
}
timeout_handler()
{
    #place here: commands to run when timeout signal (set outside to USR1) arrives
    kill -TERM ${1} #forward SIGTERM signal to the user application
    #manually requeue here because slurm *will not* do it automatically
    scontrol requeue ${SLURM_JOB_ID}
}

python make_source_map.py configs/make_source_map/nersc_atm.yaml &
pid=$!
trap "preempt_handler '$pid'" SIGTERM #this catches preempt SIGTERM from slurm
trap "timeout_handler '$pid'" USR1 #this catches timeout USR1 from slurm
wait
sleep 120 #keep the job step alive until slurm sends SIGKILL
