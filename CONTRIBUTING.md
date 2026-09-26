# Contributing

Thanks for looking. Vision is a small project, so keep changes focused and talk first (open an issue)
before anything big.

## Setup

```bash
git clone https://github.com/domdevz/vision.git && cd vision
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[all]'
```

Text-only work doesn't need the `voice` extra or a GPU.

## Tests

```bash
env -u CODEX_HOME .venv/bin/python -m unittest discover -s tests
```

The tests use fakes for the CLIs, the GPU and the microphone, so they don't touch your subscriptions.
Run them with `CODEX_HOME` unset, as above, so the Codex tests don't pick up a real config.

## Style

- Match the code around you: plain functions and dataclasses, comments that say *why*, no framework
  for its own sake.
- Anything heavy (torch, numpy, audio, the server) is imported inside the function that needs it, so
  the text chat installs and starts without the optional extras. Keep it that way.
- Don't add anything personal to the repo: hostnames, IPs, keys, home paths, locations.
- New settings go in `DEFAULT_CONFIG` in `vision/config.py` with a comment, and in
  [docs/guide.md](docs/guide.md) if users will look for them.

## Licence

By contributing you agree that your work is released under the [GPL-3.0-or-later](LICENSE).
