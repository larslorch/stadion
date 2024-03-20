import os
import sys
import subprocess
import datetime

import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"


def generate_base_command(module, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list=None,
                          array_command=None, array_indices=None, array_throttle=None,
                          n_cpus=1, n_gpus=0, dry=False, only_estimate_time=False,
                          mem=3000, length=None, hours=None, mins=59,
                          mode='local', gpu_model=None, prompt=True, gpu_mtotal=None,
                          relaunch=False, relaunch_after=None, output_filename="", output_path_prefix=""):

    if only_estimate_time:
        dry = False

    # check if single or array
    is_array_job = array_command is not None and array_indices is not None
    assert (command_list is not None) or is_array_job
    if is_array_job:
        assert all([(ind.isdigit() if type(ind) != int else True) for ind in array_indices]), f"array indices must be positive ints but got `{array_indices}`"

    # If you want to submit cluster jobs - you need to run this on the cluster login node
    if mode == 'cluster': #slurm
        cluster_cmds = []
        slurm_cmd = 'sbatch '

        if only_estimate_time:
            slurm_cmd += "--test-only "

        # Wall-clock time  hours:minutes:secs
        if length == "very_long":
            slurm_cmd += f'--time=119:{mins}:00 '
        elif length == "long":
            slurm_cmd += f'--time=23:{mins}:00 '
        elif length == "short":
            slurm_cmd += f'--time=3:{mins}:00 '
        elif length is None:
            assert hours is not None and int(hours) >= 0
            slurm_cmd += f'--time={int(hours)}:{mins}:00 '
        else:
            raise NotImplementedError(f"length `{length}` not implemented")

        # CPU memory and CPUs
        slurm_cmd += f'-n {n_cpus} ' # Number of CPUs
        slurm_cmd += f'--mem-per-cpu={mem} '

        # GPU
        if n_gpus > 0:
            if type(gpu_model) == list:
                raise NotImplementedError("pass single gpu specifier correct gpu specifier")

            gpu_model_spec = f'{gpu_model}:' if gpu_model is not None else ""
            slurm_cmd += f'--gpus={gpu_model_spec}{n_gpus} '

            if gpu_mtotal is not None:
                slurm_cmd += f'--gres=gpumem:{gpu_mtotal} ' # makes sure to select GPU with at least this MB memory

        if is_array_job:
            command_list = [array_command]

        for python_cmd in command_list:

            # add job descr
            if "--descr" in python_cmd:
                job_descr = python_cmd.split("--descr ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            elif "--run_name" in python_cmd:
                job_descr = python_cmd.split("--run_name ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            else:
                warnings.warn("--run_name/--descr/--name not in python cmd, generating own `-J` slurm job description that could be odd")
                job_descr = python_cmd.split(".py", 1)[1].replace(" ", "_").replace("--", "")

            job_descr = job_descr.replace("\$SLURM_ARRAY_TASK_ID", "%a")
            slurm_cmd_run = slurm_cmd
            slurm_cmd_run += f'-J "{output_filename}{job_descr}" '
            slurm_cmd_run += f'-o "{output_path_prefix}slurm-{output_filename}{job_descr}.txt" '

            if is_array_job:
                if type(array_indices) == range:
                    slurm_cmd_run += f'--array {array_indices.start}-{array_indices.stop - 1}'
                    if array_throttle is not None:
                        slurm_cmd_run += f"%{array_throttle}"
                else:
                    slurm_cmd_run += f'--array {",".join([str(ind) for ind in array_indices])}'

            # add relaunch
            if not relaunch:
                cluster_cmds.append(slurm_cmd_run + " --wrap \"" + python_cmd + "\"")
            else:
                relaunch_flags = f" --relaunch True " \
                                 f" --relaunch_str \'" + slurm_cmd_run.replace("\"", "\\\"") + "\' "
                if relaunch_after is None:
                    relaunch_flags += f' --relaunch_after {60 * (119 if length == "very_long" else (23 if length == "long" else 3))} '
                else:
                    relaunch_flags += f' --relaunch_after {relaunch_after} '

                # add datetime to slurm command only after relaunch flags are set
                slurm_cmd_run = slurm_cmd_run.replace(".out", f"_{datetime.datetime.now().strftime('%d-%m-%H:%M')}.out")
                cluster_cmds.append(slurm_cmd_run + " --wrap \"" + python_cmd + relaunch_flags  +"\"")

        if prompt and not dry:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            print()
            for cmd in cluster_cmds:
                if dry:
                    print(cmd, end="\n")
                else:
                    print(cmd, end="\n")
                    os.system(cmd)

    # deprecated LSF batching system
    elif mode == 'cluster_lsf':
        cluster_cmds = []
        bsub_cmd = (
            'bsub '
            + f'-W {119 if length == "very_long" else (23 if length == "long" else 3)}:58 '  # Wall-clock time  hours:minutes
            + f'-R "rusage[mem={mem}]" ' # Memory
            + (f'-R "rusage[ngpus_excl_p={n_gpus}]" ' if n_gpus > 0 else "") # Number of GPUs
            + f'-n {n_cpus} ' # Number of CPUs
            + f'-R "span[hosts={1}]" '
        )
        if gpu_model is not None:
            if isinstance(gpu_model, list) or isinstance(gpu_model, tuple):
                gpu_model_strg = " || ".join([f"gpu_model0=={gpu_}" for gpu_ in gpu_model])
                bsub_cmd += f'-R "select[({gpu_model_strg})]" '  # specific GPU model
            else:
                bsub_cmd += f'-R "select[gpu_model0=={gpu_model}]" '  # specific GPU model
        if gpu_mtotal is not None:
            bsub_cmd += f'-R "select[gpu_mtotal0>={gpu_mtotal}]" ' # makes sure to select GPU with at least this memory

        if is_array_job:
            command_list = [array_command]

        for python_cmd in command_list:

            # add job descr
            job_descr = python_cmd.split("--run_name ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            bsub_cmd_run = bsub_cmd
            if is_array_job:
                if type(array_indices) == range:
                    arr_inpt = f"{array_indices.start}-{array_indices.stop - 1}"
                else:
                    arr_inpt = ",".join([str(ind) for ind in array_indices])
                bsub_cmd_run += f'-J "{output_filename}{job_descr}[{arr_inpt}]" '
                bsub_cmd_run += f'-o "{output_path_prefix}lsf.o-{output_filename}{job_descr}.txt" '
            else:
                bsub_cmd_run += f'-J "{output_filename}{job_descr}" '
                bsub_cmd_run += f'-o "{output_path_prefix}lsf.o-{output_filename}{job_descr}.txt" '

            if not relaunch:
                cluster_cmds.append(bsub_cmd_run + "\"" + python_cmd + "\"")
            else:
                relaunch_flags = f" --relaunch True " \
                                 f" --relaunch_str \'" + bsub_cmd_run.replace("\"", "\\\"") + "\' "
                if relaunch_after is None:
                    relaunch_flags += f' --relaunch_after {60 * (119 if length == "very_long" else (23 if length == "long" else 3))} '
                else:
                    relaunch_flags += f' --relaunch_after {relaunch_after} '

                cluster_cmds.append(bsub_cmd_run + "\"" + python_cmd + relaunch_flags  +"\"")

        if prompt and not dry:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                if dry:
                    print(cmd, end="\n\n")
                else:
                    os.system(cmd)

    elif mode == 'local':
        if prompt and not dry:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if is_array_job:
            command_list = [array_command.replace("\$SLURM_ARRAY_TASK_ID", str(ind)) for ind in array_indices]

        if answer == 'yes':
            print()
            for cmd in command_list:
                if dry:
                    print(cmd, end="\n")
                else:
                    subprocess.call(cmd, shell=True)

    else:
        raise NotImplementedError

