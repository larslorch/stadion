import subprocess
import sys
import argparse
from datetime import datetime

def get_commit_datetime(commit_hash=None, date_format="format:%m_%d-%H_%M"):
    if commit_hash is None:
        commit_hash = get_version()
    return subprocess.check_output(["git", "log", "-n", "1", f"--format=%cd", f"--date={date_format}", commit_hash]).strip().decode("utf-8")


def get_version():
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode(sys.stdout.encoding)
    return git_commit


def get_datetime():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H%M")
    return date_time


def get_gpu_info():
    try:
        sp = subprocess.Popen(['lspci'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()
        out_list = out_str[0].decode().split('\n')

        out_gpu = [x for x in out_list if 'VGA' in x]
        return out_gpu
    except FileNotFoundError as e:
        return "No GPU info"


def get_gpu_info2():
    try:
        line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        line = line_as_bytes.decode("ascii")
        _, line = line.split(":", 1)
        line, _ = line.split("(")
        return line.strip()
    except subprocess.CalledProcessError as e:
        return "No GPU info"


def str2bool(v):
    v = "".join([char for char in v if u"\xa0" not in char])  # avoid utf8 parsing error
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 'T', 't', 'Y', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'F', 'f', 'N', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_descr(kwargs, skip_fields=None, short=False):
    skip_fields = skip_fields or []

    if "config" in kwargs:
        config = kwargs["config"].copy()
        del kwargs["config"]
        kwargs.update(config)

    if short:
        descr = "-".join(sorted([
            "".join([c[0] for c in f"{k}".split("_")]) + f"={v}" for k, v in kwargs.items() if k not in skip_fields
        ]))
    else:
        # descr = "-".join(sorted([
        #     "".join([c for c in f"{k}".split("_")]) + f"={v}" for k, v in kwargs.items() if k not in skip_fields
        # ]))
        descr = "-".join(sorted([
            f"{k}={v}" for k, v in kwargs.items() if k not in skip_fields
        ]))
    return descr