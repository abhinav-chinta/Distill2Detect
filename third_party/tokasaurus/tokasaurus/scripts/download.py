import huggingface_hub
import pydra


class ScriptConfig(pydra.Config):
    model: str

    def __init__(self):
        super().__init__()
        self.allow_patterns = ["*.safetensors", "*.json"]


def download(config: ScriptConfig):
    print(f"Downloading {config.model}")
    cached_path = huggingface_hub.snapshot_download(
        config.model, allow_patterns=config.allow_patterns
    )
    print(f"Download complete, stored at {cached_path}")


def main():
    pydra.run(download)


if __name__ == "__main__":
    main()
