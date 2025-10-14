import subprocess
import shlex

scripts = [
    "uv run main.py --model unetpp --image_size 512 --batch_size 8 -wandb true --include_classes 4",
    "uv run main.py --model unet   --image_size 512 --batch_size 8 -wandb true --include_classes 4",
    "uv run main.py --model unetpp --image_size 512 --batch_size 8 -wandb true --include_classes 1 4",
    "uv run main.py --model unet   --image_size 512 --batch_size 8 -wandb true --include_classes 1 4",
    "uv run main.py --model unetpp --image_size 512 --batch_size 8 -wandb true --include_classes 2 4",
    "uv run main.py --model unet   --image_size 512 --batch_size 8 -wandb true --include_classes 2 4",
    "uv run main.py --model unetpp --image_size 512 --batch_size 8 -wandb true --include_classes 3 4",
    "uv run main.py --model unet   --image_size 512 --batch_size 8 -wandb true --include_classes 3 4",
]


def run_scripts(cmds, stop_on_error=True):
    for i, cmd in enumerate(cmds, 1):
        print(f"\n[{i}/{len(cmds)}] Running: {cmd}")
        try:
            # On Windows, set shell=True if 'uv' isn't found on PATH resolution
            result = subprocess.run(shlex.split(cmd), check=True)
        except FileNotFoundError:
            # Fallback for Windows/certain shells where 'uv' needs shell=True
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0 and stop_on_error:
                raise SystemExit(f"Command failed with code {result.returncode}: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed (exit {e.returncode}): {cmd}")
            if stop_on_error:
                raise
        else:
            print(f"✅ Done: {cmd}")


run_scripts(scripts)
