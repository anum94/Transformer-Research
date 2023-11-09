import argparse
import os
import json


def config_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to generate and run necessary execution files"
    )
    parser.add_argument(
        "--exec",
        type=str,
        choices=["summ", "eval", "llm"],
        help="Type of execution: summ or eval or llm eval",
        required=True,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=[
            "lrz-dgx-a100-80x8",
            "lrz-v100x2",
            "lrz-dgx-1-v100x8",
            "lrz-dgx-1-p100x8",
            "lrz-hpe-p100x4",
        ],
        help="GPU partition to use",
        required=True,
    )
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use", default=1)
    parser.add_argument(
        "--max_time", type=int, help="Maximum time for execution in minutes", default=60
    )
    parser.add_argument(
        "--dsconfig",
        type=int,
        choices=[0, 2, 3],
        help="Deepspeed optimization configuration",
        default=0,
    )
    parser.add_argument(
        "--ckpt", type=str, help="Whether to resume training from a checkpoint"
    )

    return parser.parse_args()


def get_exec_str(args) -> str:
    if parser.exec == "llm":
        return f"{args['dataset']}-{args['model_hf_key']}"

    method = args["method"]
    model = args["exec_args"]["model"]
    dataset = args["exec_args"]["dataset"]
    enc_max_len = args["exec_kwargs"]["enc_max_len"]
    return f"{method}-{model}-{dataset}-{enc_max_len}"


if __name__ == "__main__":
    parser = config_parser()
    exec_config_path = f"{parser.exec}.json"
    ckpt_path = parser.ckpt if parser.ckpt else None

    # read run config for folder name
    with open(exec_config_path, "r") as j:
        exec_config = json.load(j)

    # get full run path from the config json file
    exec_path = get_exec_str(exec_config)
    aug_exec_path = os.path.join("lrz", "runs", exec_path)

    # mkdir aug exec path
    if not os.path.exists(aug_exec_path):
        os.makedirs(aug_exec_path)

    config_path = os.path.join(aug_exec_path, "config.json")
    with open(config_path, "w") as fp:
        json.dump(obj=exec_config, fp=fp)

    # create dump files
    dump_out_path = os.path.join(aug_exec_path, "dump.out")
    dump_err_path = os.path.join(aug_exec_path, "dump.err")
    os.system(f"touch {dump_err_path}")
    os.system(f"touch {dump_out_path}")

    og_path_container = "/dss/dssfs04/lwp-dss-0002/t12g1/t12g1-dss-0000/"

    # create sbatch file
    sbatch_path = os.path.join(aug_exec_path, "run.sbatch")
    with open(sbatch_path, "w") as sbatch_file:
        sbatch_file.write("#!/bin/bash\n")
        sbatch_file.write("#SBATCH -N 1\n")
        sbatch_file.write(f"#SBATCH -p {parser.gpu}\n")
        sbatch_file.write(f"#SBATCH --gres=gpu:{parser.num_gpus}\n")
        sbatch_file.write("#SBATCH --ntasks=1\n")
        sbatch_file.write(f"#SBATCH -o {dump_out_path}\n")
        sbatch_file.write(f"#SBATCH -e {dump_err_path}\n")
        sbatch_file.write(f"#SBATCH --time={parser.max_time}\n\n")

        if parser.exec == "llm":
            srun_command = f"srun --container-image ~/demo.sqsh --container-mounts={og_path_container}:/mnt/container torchrun --nproc_per_node={parser.num_gpus} --standalone ~/transformer-research/llm_summ_eval.py --config ~/transformer-research/{config_path}"

        elif parser.dsconfig == 0:
            srun_command = f"srun --container-image ~/demo.sqsh --container-mounts={og_path_container}:/mnt/container torchrun --nproc_per_node={parser.num_gpus} --standalone ~/transformer-research/main.py --config ~/transformer-research/{config_path}"

        else:
            srun_command = f"srun --container-image ~/demo.sqsh --container-mounts={og_path_container}:/mnt/container deepspeed --num_gpus={parser.num_gpus} --no_local_rank ~/transformer-research/main.py --config ~/transformer-research/{config_path} --dsconfig ~/transformer-research/dsconfig_zero{parser.dsconfig}.json {'--ckpt ' + ckpt_path if ckpt_path else ''}"
        sbatch_file.write(f"{srun_command}\n")

    # submit sbatch job
    os.system(f"sbatch {sbatch_path}")
