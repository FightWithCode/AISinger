# 🎤 AISinger

Make **any voice sing any song**: your own voice singing *O Mere Dil Ke Chain*, a Mohammed Rafi song in Kishore Kumar's voice, and so on.

AISinger doesn't generate singing from scratch, which is what makes AI vocals sound robotic. It takes a **real performance** and changes only the **voice** (singing voice conversion). Melody, timing, breathing, vibrato and emotion all come from a human singer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FightWithCode/AISinger/blob/main/notebooks/AISinger_Colab.ipynb)

## How it works

```
song.mp3 ──► separate ──► vocals ──► lead / backing ──► de-reverb ──► VOICE CONVERSION ──► mix ──► cover.mp3
              (BS-RoFormer)        (Mel-RoFormer karaoke)  (UVR DeEcho)   RVC  or  Seed-VC     (level-match,
                 └──► instrumental ─────────────────────────────────────────────────────────►  reverb, key shift)
```

| Part | Tool | Why |
|---|---|---|
| Stem separation | [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (UVR models) | State-of-the-art RoFormer models; clean vocals matter more than anything else |
| Voice cloning (trained) | [Applio](https://github.com/IAHispano/Applio) (RVC v2) | Most natural results once trained on 10–30 min of the voice |
| Voice cloning (instant) | [Seed-VC](https://github.com/Plachtaa/seed-vc) | Zero-shot from a 10–25 s clip; optional fine-tuning |
| Mixing | [pedalboard](https://github.com/spotify/pedalboard) | Level matching, compression, reverb, high-quality pitch shift |
| Speech in a cloned voice | Applio TTS (edge-tts → RVC) | Talk in the cloned voice, too |

Each engine is installed at a pinned version into its **own virtualenv** (using `uv`), so their conflicting dependencies never break each other or Colab.

## Quick start: Google Colab (recommended, free GPU)

1. Click **Open in Colab** above. Choose `Runtime → Change runtime type → T4 GPU`.
2. Run **Step 1** (about 5–8 minutes the first time in a session). It mounts Drive, installs AISinger from this GitHub repo, and sets up the engines.
3. Use the sections: **A** clone (train) a voice · **B** instant voice from a clip · **C** import a model · **D** make a cover · **E** convert a vocal · **F** speak text.

Everything you create (voices, datasets, covers) is saved to `MyDrive/AISinger`, so training resumes after a Colab disconnect.

## Install anywhere (Linux / WSL / macOS with an NVIDIA GPU)

```bash
pip install "aisinger[gpu] @ git+https://github.com/FightWithCode/AISinger"
```

```bash
aisinger setup          # installs the RVC and Seed-VC engines (add --cpu for no GPU)
```

```bash
aisinger doctor
```

Requires Python ≥ 3.10, `git` and `ffmpeg`. Data goes to `~/AISinger` and caches to `~/.cache/aisinger`. Override them with `AISINGER_HOME` and `AISINGER_CACHE`.

## Usage

### 1. Clone your voice (best quality)

```bash
aisinger dataset me ~/recordings/            # your a-cappella singing, 10–30 min total
```

```bash
aisinger train me --epochs 300               # ~20–60 min on a T4, saves every 25 epochs
```

```bash
aisinger cover "O Mere Dil Ke Chain.mp3" --voice me --key -2
```

### 2. Clone a famous singer from their songs

```bash
aisinger dataset kishore kishore_songs/ --isolate --lead-only --dereverb
```

```bash
aisinger train kishore
```

```bash
aisinger cover rafi_song.mp3 --voice kishore
```

### 3. Instant voice (no training)

```bash
aisinger reference kishore_zs clip.mp3 --isolate --start 30   # 25 s starting at 0:30
```

```bash
aisinger cover rafi_song.mp3 --voice kishore_zs --engine seedvc
```

### 4. Other commands

```bash
aisinger convert my_singing.wav --voice kishore          # vocal-only file → another voice
aisinger speak "Namaste, kaise ho?" --voice me           # text → speech in your voice
aisinger import kishore https://huggingface.co/.../model.zip   # use an existing RVC model
aisinger use me                                          # list saved epochs
aisinger use me me_200e_4800s.pth                        # pick an earlier, less over-trained epoch
aisinger voices
```

### Cover options

| Option | Meaning |
|---|---|
| `--pitch N` | Shift **only the vocal** (`+12` male→female, `-12` female→male). Keeps the song's key. |
| `--key N` | Transpose **the whole song** (vocal and music) into the new singer's comfortable range. |
| `--backing keep/convert/drop` | What to do with chorus and backing vocals. |
| `--index-rate 0–1` | (RVC) How much of the target's tone and accent to impose. 0.5 is a good start; lower it if you hear artifacts. |
| `--protect 0–0.5` | (RVC) Protects breaths and consonants. Lower it if "s"/"t" sounds turn robotic. |
| `--reverb 0–1` | Reverb added to the new vocal (the original reverb is removed first). |
| `--vocal-gain dB` | Make the vocal louder or quieter in the mix. |

## Getting natural, non-robotic results

1. **Training data is everything.** Use dry singing (no music, no echo), 10–30 minutes, covering your full range. Phone recordings in a quiet room with soft furnishings are fine.
2. **Don't over-train.** If the latest epoch sounds metallic, try an earlier one with `aisinger use`.
3. **Match ranges.** If the source melody is far outside the target voice's range, use `--key` to transpose the song. Pushing a voice an octave beyond its range sounds fake.
4. **Better input, better output.** Old mono recordings (1960s–70s Bollywood) separate imperfectly. For your own voice, singing the song yourself and converting that (`aisinger convert`) often beats converting the original.
5. **Try both engines.** RVC is usually cleaner. Seed-VC can capture timbre better from little data.

## Ethics & legal

Clone only voices you have permission to use. Personal experiments with covers are common, but **publishing** an AI cover of a real singer can violate personality rights (Indian courts have protected singers' voices against AI cloning) and the song's copyright. Label AI-generated audio as such.

## Credits & licenses

AISinger's own code is MIT-licensed. It downloads and runs, unmodified, [Applio](https://github.com/IAHispano/Applio) (MIT), [Seed-VC](https://github.com/Plachtaa/seed-vc) (GPL-3.0), [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT) and UVR community models. Their licenses and terms of use apply to them.
