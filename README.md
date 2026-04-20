# vlm-annotate-verify

VLM-proposed robot episode annotation with terminal UI verification.

```
propose <root>   →   review <root>   →   export <root>
   ~12s/1K eps        ~4s/ep human         <1s
```

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
vlm-annotate-verify propose ./my_dataset
vlm-annotate-verify review ./my_dataset
vlm-annotate-verify export ./my_dataset
```

## Requirements

- Python 3.12+
- ffmpeg on PATH
- `GEMINI_API_KEY` in environment
- A terminal that supports sixel or Kitty graphics protocol (e.g. WezTerm)
