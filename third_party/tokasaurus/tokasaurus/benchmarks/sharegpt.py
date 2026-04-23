import subprocess

import pydra

from tokasaurus.benchmarks.utils import (
    BaseConfig,
    launch_server,
    prepend_conda_activate,
)


class Config(BaseConfig):
    sharegpt_command: str
    sharegpt_env: str | None = None


def main(config: Config):
    sharegpt_command = config.sharegpt_command

    if config.save_path is not None:
        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        sharegpt_command = f"{sharegpt_command} --output-file {config.save_path}"

    if config.sharegpt_env is not None:
        sharegpt_command = prepend_conda_activate(
            sharegpt_command, config.conda_activate_path, config.sharegpt_env
        )

    if (save_path := config.save_path) is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        launch_command_save_path = save_path.with_suffix(".launch.txt")
        launch_command_save_path.write_text(str(config.launch))

    print(f"ShareGPT command: '{sharegpt_command}'")

    with launch_server(config):
        for _ in range(config.reps):
            subprocess.run(sharegpt_command, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    pydra.run(main)
