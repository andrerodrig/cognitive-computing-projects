from pathlib import Path

def root_path() -> Path:
    current_path = Path.cwd()
    while not (current_path / 'pyproject.toml').is_file():
        current_path = current_path.parent
    return current_path


if __name__ == '__main__':
    print('\n', root_path())