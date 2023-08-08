import argparse
import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run command lines with varying desc, lim, and reason.")
    parser.add_argument("--envs", type=int, help="Specify the envs ID (3 to 10).")
    parser.add_argument("--desc-start", type=int, help="Specify the start value for desc.")
    parser.add_argument("--desc-end", type=int, help="Specify the end value for desc (included).")
    args = parser.parse_args()

    if not (0 <= args.envs <= 10):
        print("Error: envs ID should be between 3 and 10 (inclusive).")
    elif args.desc_start is None or args.desc_end is None:
        print("Error: Both --desc-start and --desc-end must be provided.")
    elif args.desc_start > args.desc_end:
        print("Error: --desc-start should be less than or equal to --desc-end.")
    else:
        for value in range(args.desc_start, args.desc_end + 1, 50):
            command = f"python train.py --log --steps 100 --desc {value} --lim {value} --memo 10 --reason {value} --wandb --envs {args.envs}"
            run_command(command)
