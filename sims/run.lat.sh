#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --qos=regular
#SBATCH --account=mp107b

sso_name=Mars

# Pick a tube and figure out the offset from boresight
tele="LAT"
tubes=(c1 i1 i3 i4 i5 i6 o6)
days=(15 16) #17 18 19 20 21 22)
beam_dir=/global/cfs/cdirs/sobs/bcp
export OMP_NUM_THREADS=16
ntask=8
# export TOAST_LOGLEVEL=VERBOSE

for tube in "${tubes[@]}";
do
    case $tube in
        o6)
        bands=(f030 f040)
    	;;
        i1|i3|i4|i6)
        bands=(f090 f150)
    	;;
        c1|i5)
        bands=(f230 f290)
    	;;
        *)
    	echo "2 Unknown band: $band"
    	exit
    	;;
    esac
    # NOTE:  for wafer offsets, use the --wafer_slot option
    offsets=(`get_wafer_offset --tube_slots $tube`)
    offset_az=${offsets[0]}
    offset_el=${offsets[1]}
    tube_radius=${offsets[2]}
    
    for band_name in "${bands[@]}";
    do
        echo "tube ${tube} band ${band_name}"
        echo "offset_az = ${offset_az}"
        echo "offset_el = ${offset_el}"
        echo "tube_radius = ${tube_radius}"
        
        beam_file=${beam_dir}/${tele}_${band_name}_beam.h5
        
        for i in "${days[@]}";
        do
            # Simulate a scanning strategy
            t_start="2025-03-$i 00:00:00"
            t_stop="2025-03-$(($i+1)) 00:00:00"
            prefix=${t_start}_${band_name}_${tube}_${sso_name}
            prefix="${prefix// /_}"
            
            outdir="/global/cfs/cdirs/sobs/users/skh/sso_sims/${prefix}"
            echo ${outdir}
            
            mkdir -p "${outdir}"
            schedule=${outdir}/schedule_${tube}_${sso_name}.txt
            
            toast_ground_schedule \
                --site-lat -22.958064 \
                --site-lon -67.786222 \
                --site-alt 5200 \
                --site-name ATACAMA \
                --telescope ${tele} \
                --patch-coord C \
                `#--elevations-deg "59.0"` \
                --el-min-deg 40 \
                --el-max-deg 65 \
                --sun-el-max-deg 90 \
                --sun-avoidance-angle-deg 45 \
                --moon-avoidance-angle-deg 45 \
                --start "$t_start" \
                --stop "$t_stop" \
                --gap-s 86400 \
                --gap-small-s 0 \
                --ces-max-time-s 1200 \
                --fp-radius-deg 0 \
                --patch ${sso_name},SSO,1,${tube_radius} \
                --boresight-offset-az-deg ${offset_az} \
                --boresight-offset-el-deg ${offset_el} \
                --out "$schedule"
            
            # Run a simulation
            case $band_name in
                f030|f040)
            	export_key=tube_slot
            	fsample=200
            	;;
                f090|f150)
            	export_key=wafer_slot
            	fsample=200
            	;;
                f230|f290)
            	export_key=wafer_slot
            	fsample=200
            	;;
                *)
            	echo "2 Unknown band: $band"
            	exit
            	;;
            esac
            
            # NOTE:  These parameters can also be set with a config file...
            # The "names" of each operator in the command line arguments are just
            # the names given in the script being called (toast_so_sim in this
            # case).
            srun -n ${ntask} toast_so_sim \
                `#--corotate_lat.no_corotate_lat` \
                `# Do not make a map, just write the timestreams` \
                --mapmaker.disable \
                `# Instrument params` \
                --tube_slots ${tube} \
                --bands LAT_${band_name} \
                --sample_rate "${fsample}" \
                `# Observing schedule` \
                --schedule ${schedule} \
                `# Scanning params` \
                --sim_ground.turnaround_mask 2 \
                --sim_ground.scan_rate_az '1.5 deg / s' \
                --sim_ground.scan_accel_az '3.0 deg / s2' \
                `# Use fixed weather parameters` \
                --sim_ground.median_weather \
                `# Simulated sky signal from a map` \
                --scan_map.disable \
                `# Simulated SSO` \
                --sim_sso.enable \
                --sim_sso.sso_name ${sso_name} \
                --sim_sso.beam_file ${beam_file} \
                `# Simulated atmosphere params (high resolution)` \
                --sim_atmosphere.enable \
                --sim_atmosphere.field_of_view '6 deg' \
                --sim_atmosphere.cache_dir '/global/cfs/cdirs/sobs/users/skh/toast_cache' \
                `# Simulated atmosphere params (coarse resolution)` \
                `# --sim_atmosphere_coarse.enable` \
                `# --sim_atmosphere_coarse.field_of_view '6 deg'` \
                `# Noise simulation (from elevation-modulated focalplane parameters)` \
                --sim_noise.enable \
                `# Gain mismatch` \
                `#--gainscrambler.enable` \
                `#--gainscrambler.sigma 0.01` \
                `# Timeconstant convolution` \
                `#--convolve_time_constant.disable` \
                `#--convolve_time_constant.tau '3 ms'` \
                `# Write to HDF5` \
                --save_hdf5.enable \
                --out_dir ${outdir} \
                --job_group_size ${ntask} | tee "${outdir}/log"
        done
    done
done
