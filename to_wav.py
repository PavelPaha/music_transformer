import argparse
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from midi2audio import FluidSynth

def sine_note(freq, duration, velocity, fs, 
              attack=0.01, release=0.05):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    amp = velocity / 127.0
    env = np.ones_like(wave) * amp
    a_samples = int(fs * attack)
    r_samples = int(fs * release)
    if a_samples > 0:
        env[:a_samples] = np.linspace(0, amp, a_samples)
    if r_samples > 0:
        env[-r_samples:] = np.linspace(amp, 0, r_samples)
    return wave * env


def midi_to_wav_puresynth(midi_path: str, wav_path: str, fs: int = 44100):
    pm = pretty_midi.PrettyMIDI(midi_path)
    total_duration = pm.get_end_time()
    audio = np.zeros(int(fs * (total_duration + 1)), dtype=np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            f = pretty_midi.note_number_to_hz(note.pitch)
            start_idx = int(fs * note.start)
            wave = sine_note(f, note.end - note.start, note.velocity, fs)
            end_idx = start_idx + wave.shape[0]
            if end_idx > len(audio):
                wave = wave[:len(audio)-start_idx]
                end_idx = len(audio)
            audio[start_idx:end_idx] += wave

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, fs)
    print(f"ok")

def midi_to_wav_soundfont(midi_path: str, wav_path: str, sf2_path: str, fs: int = 44100):
    pm = pretty_midi.PrettyMIDI(midi_path)
    print_midi_content(pm)
    fsynth = FluidSynth(sound_font=sf2_path, sample_rate=fs)
    fsynth.midi_to_audio(midi_path, wav_path)
    print(f"ok")

def main():
    parser = argparse.ArgumentParser(
        description="MIDI → WAV: чистый синтез или через SoundFont"
    )
    parser.add_argument("midi_file", help="Входной MIDI-файл (.mid)")
    parser.add_argument("wav_file",  help="Выходной WAV-файл (.wav)")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Частота дискретизации (по умолчанию 44100)")
    parser.add_argument("--sf2", type=str, default=None,
                        help="(опц.) Путь к .sf2-файлу для рендера через FluidSynth")
    args = parser.parse_args()

    if args.sf2:
        midi_to_wav_soundfont(args.midi_file, args.wav_file, args.sf2, fs=args.sr)
    else:
        midi_to_wav_puresynth(args.midi_file, args.wav_file, fs=args.sr)

if __name__ == "__main__":
    main()
