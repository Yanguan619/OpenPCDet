import subprocess
from pathlib import Path



def get_git_commit_number():
    if not (Path(__file__).parent / '../.git').exists():
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


script_version = get_git_commit_number()
