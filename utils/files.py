import os
import git
import logging
from json import dump
from torch.cuda import current_device, is_available


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def log_output(path, output) -> None:
    output = map(lambda x: x + "\n", output)
    with open(path, "w") as fp:
        fp.writelines(output)


def log_json(path, metrics) -> None:
    with open(path, "w") as fp:
        dump(metrics, fp)


def mkdir(path) -> None:
    if is_available() and current_device() == 0:
        os.mkdir(path)


def push_files_git(
    commit_msg: str, file_paths: list, repo_path: str, ssh_key_path: str
) -> None:
    try:
        ssh_cmd = f"ssh -o 'StrictHostKeyChecking=no' -i {ssh_key_path}"
        repo = git.Repo(path=repo_path)
        repo.index.add(file_paths)
        repo.index.commit(commit_msg)
        repo.remotes.origin.push()
        logging.info(
            f"Results were commited to Gitlab with commit message: {commit_msg}"
        )
    except Exception as e:
        logging.error(f"Failed to commit results to Gitlab.\nException message: {e}")
