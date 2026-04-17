# audio_ml

Local audio processing utilities for multi-speaker conversation transcription.

The `input_audio_processing` package loads a local audio file, detects speech regions, extracts speaker embeddings,
optionally matches speech against an enrolled user voice sample, clusters the remaining speakers, transcribes speech with
Whisper, and emits speaker-attributed transcript turns.

## What It Does

- Loads audio with `soundfile`, converts it to mono, resamples it to 16 kHz, and normalizes the waveform.
- Detects speech with Silero VAD.
- Splits speech into embedding-friendly segments.
- Extracts speaker embeddings with SpeechBrain ECAPA-TDNN.
- Optionally labels segments matching a known `USER` voice sample.
- Groups unknown speakers with one of three strategies:
  - `unique_speaker_similarity`
  - `centroid_similarity`
  - `agglomerative`
- Transcribes ASR chunks with local Whisper word timestamps.
- Aligns timestamped words back to speaker segments.
- Prints readable transcript turns and writes JSON output.

## Project Layout

```text
input_audio_processing/
  audio_processor.py      Audio loading, normalization, VAD segmentation, and ASR chunking
  cli.py                  Command-line entry point
  config.py               Pipeline configuration dataclass
  embedding_extractor.py  Speaker embedding extraction
  pipeline.py             End-to-end orchestration
  speaker_clusterer.py    Unknown-speaker grouping
  speaker_identifier.py   Enrolled-user matching
  transcriber.py          Whisper transcription wrapper
  types.py                Shared dataclasses
  models/                 Local model wrappers
test/
  voice_processing_demo.ipynb
  youtube_audio.wav
  youtube_downloaded_audio.m4a
```

## Requirements

Python 3.11 was used for verification in this workspace.

Install the runtime dependencies:

```bash
python3 -m pip install numpy torch torchaudio soundfile silero-vad speechbrain openai-whisper certifi
```

Whisper also needs `ffmpeg` available on your system PATH.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

The first run may download model weights for Silero VAD, SpeechBrain ECAPA, and Whisper.

## Usage

Run clustering-only diarization and transcription:

```bash
python3 -m input_audio_processing.cli test/youtube_audio.wav \
  --output-json voice_transcript.json
```

Run with an enrolled user voice sample:

```bash
python3 -m input_audio_processing.cli path/to/conversation.wav path/to/user_voice.wav \
  --output-json voice_transcript.json
```

Choose a Whisper model and language:

```bash
python3 -m input_audio_processing.cli path/to/conversation.wav \
  --whisper-model small \
  --language en \
  --output-json voice_transcript.json
```

Tune speaker grouping:

```bash
python3 -m input_audio_processing.cli path/to/conversation.wav \
  --speaker-strategy unique_speaker_similarity \
  --speaker-threshold 0.72 \
  --cluster-threshold 0.4
```

## CLI Options

```text
audio_path                         Source conversation audio file
user_voice_path                    Optional enrolled USER voice sample
--output-json                      JSON transcript path, default: voice_transcript.json
--whisper-model                    Whisper model name, default: small
--language                         Optional language code passed to Whisper
--user-threshold                   USER similarity threshold, default: 0.7
--speaker-strategy                 unique_speaker_similarity, centroid_similarity, or agglomerative
--speaker-threshold                Speaker similarity threshold, default: 0.72
--cluster-threshold                Agglomerative clustering distance threshold, default: 0.4
--vad-threshold                    Silero VAD threshold, default: 0.5
--min-segment-seconds              Minimum speech segment length, default: 0.6
--max-segment-seconds              Maximum speech segment length, default: 3.0
```

## Output

The pipeline prints transcript turns:

```text
[0.40-2.15] Speaker_0: Hello there.
[2.28-4.10] Speaker_1: Hi, how are you?
```

It also writes JSON with this shape:

```json
[
  {
    "start": 0.4,
    "end": 2.15,
    "speaker": "Speaker_0",
    "text": "Hello there."
  }
]
```

When a `user_voice_path` is provided, matching segments are labeled as `USER`; all other voices are assigned
`Speaker_N` labels.

## Python API

```python
from pathlib import Path

from input_audio_processing import PipelineConfig, VoiceProcessingPipeline

config = PipelineConfig(
    whisper_model_name="small",
    language="en",
    output_json_path=Path("voice_transcript.json"),
)

pipeline = VoiceProcessingPipeline(config=config)
turns = pipeline.run("test/youtube_audio.wav")
```

## Verification

The following checks pass in this workspace:

```bash
python3 -m compileall input_audio_processing
python3 - <<'PY'
import input_audio_processing
print("import input_audio_processing: ok")
PY
python3 -m input_audio_processing.cli --help
```

Full audio processing was not run during verification because `silero-vad` and `speechbrain` are not currently installed
in this environment.
