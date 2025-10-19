import numpy as np
import sounddevice as sd
import time
import random

def generate_tone(frequency, duration, sample_rate=44100, waveform="sine"):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if waveform == "sine": tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    elif waveform == "square": tone = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "triangle": tone = 0.5 * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
    elif waveform == "saw": tone = 0.5 * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
    elif waveform == "noise": tone = np.random.normal(0, 0.3, len(t))
    else: tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    return tone

def apply_envelope(signal, attack=0.01, decay=0.1, sustain=0.5, release=0.1, sample_rate=44100):
    n = len(signal)
    total_env_time = attack + decay + release
    if total_env_time > n / sample_rate:
        scale = (n / sample_rate) / (total_env_time + 1e-6)
        attack, decay, release = attack * scale, decay * scale, release * scale
    a, d, r = int(attack * sample_rate), int(decay * sample_rate), int(release * sample_rate)
    s = max(0, n - (a + d + r))
    a, d, r = max(1, a), max(1, d), max(1, r)
    envelope = np.concatenate([np.linspace(0, 1, a), np.linspace(1, sustain, d), np.ones(s) * sustain, np.linspace(sustain, 0, r)])
    if len(envelope) < n: envelope = np.pad(envelope, (0, n - len(envelope)), mode='constant')
    else: envelope = envelope[:n]
    return signal * envelope

def generate_alarm(melody_contour, rhythm_pattern, tempo_bpm, base_freq, env, waveform="sine"):
    sample_rate = 44100
    beat_sec = 60 / tempo_bpm
    full_signal = np.array([], dtype=np.float32)
    for i, semitone_shift in enumerate(melody_contour):
        if semitone_shift is None: continue # Skip rests for frequency calculation
        freq = base_freq * (2 ** (semitone_shift / 12))
        duration = rhythm_pattern[i % len(rhythm_pattern)] * beat_sec
        tone = generate_tone(freq, duration, waveform=waveform, sample_rate=sample_rate)
        tone = apply_envelope(tone, **env, sample_rate=sample_rate)
        full_signal = np.concatenate([full_signal, tone])
    max_amp = np.max(np.abs(full_signal))
    if max_amp > 0: full_signal /= max_amp
    return full_signal

# --- [NEW: Generative Model - Markov Chain] ---

class MarkovMelodyGenerator:
    def __init__(self):
        self.transition_matrix = {}
        self.initial_notes = []

    def train(self, melodies):
        """ Trains the model on a list of melody_contour lists. """
        for melody in melodies:
            if not melody: continue
            # Add the first note of each melody to a list of possible starting points
            self.initial_notes.append(melody[0])
            
            # Build the transition matrix
            for i in range(len(melody) - 1):
                current_note = melody[i]
                next_note = melody[i+1]
                
                if current_note not in self.transition_matrix:
                    self.transition_matrix[current_note] = {}
                
                if next_note not in self.transition_matrix[current_note]:
                    self.transition_matrix[current_note][next_note] = 0
                
                self.transition_matrix[current_note][next_note] += 1
        
        # Normalize the counts to get probabilities (optional, but good practice)
        for current_note, next_notes in self.transition_matrix.items():
            total = sum(next_notes.values())
            for next_note, count in next_notes.items():
                self.transition_matrix[current_note][next_note] = count / total

    def generate(self, length=8):
        """ Generates a new melody_contour of a given length. """
        if not self.transition_matrix:
            raise ValueError("Model has not been trained yet.")
            
        # Start with a random note from the observed initial notes
        current_note = random.choice(self.initial_notes)
        melody = [current_note]
        
        for _ in range(length - 1):
            # If we reach a note with no known next step (end of a training melody)
            if current_note not in self.transition_matrix:
                break
            
            # Get possible next notes and their probabilities
            possible_next = list(self.transition_matrix[current_note].keys())
            probabilities = list(self.transition_matrix[current_note].values())
            
            # Choose the next note based on the learned probabilities
            next_note = random.choices(possible_next, weights=probabilities, k=1)[0]
            melody.append(next_note)
            current_note = next_note
            
        return melody


if __name__ == "__main__":
    # Our "dataset" of sounds to learn from
    sound_presets = {
        "original_alarm": {"melody_contour": [0, 2, 4, 5, 7, 5, 4, 2, 0], "rhythm_pattern": [1], "tempo_bpm": 105, "base_freq": 1318.5, "env": dict(attack=0.04, decay=0.2, sustain=0.8, release=0.4), "waveform": "saw"},
        "sci_fi_warning": {"melody_contour": [0, 7, 0, 7, 0, 7], "rhythm_pattern": [0.5], "tempo_bpm": 180, "base_freq": 880, "env": dict(attack=0.01, decay=0.1, sustain=0.1, release=0.1), "waveform": "square"},
        "retro_power_up": {"melody_contour": [0, 4, 7, 12], "rhythm_pattern": [0.25], "tempo_bpm": 160, "base_freq": 261.63, "env": dict(attack=0.005, decay=0.1, sustain=0.6, release=0.1), "waveform": "square"},
        "soothing_chimes": {"melody_contour": [12, 11, 7, 4, 0, 4, 0], "rhythm_pattern": [1.5], "tempo_bpm": 80, "base_freq": 523.25, "env": dict(attack=0.01, decay=1.0, sustain=0.0, release=1.0), "waveform": "sine"},
        "game_over_fall": {"melody_contour": [12, 8, 5, 0, -4], "rhythm_pattern": [1], "tempo_bpm": 70, "base_freq": 440, "env": dict(attack=0.01, decay=0.3, sustain=0.3, release=0.5), "waveform": "square"},
        "tension_riser": {"melody_contour": list(range(13)), "rhythm_pattern": [0.2], "tempo_bpm": 140, "base_freq": 220, "env": dict(attack=0.5, decay=0, sustain=1.0, release=0.1), "waveform": "saw"},
    }

    print("Training Markov Chain model on preset melodies...")
    melody_corpus = [p["melody_contour"] for p in sound_presets.values()]
    generator = MarkovMelodyGenerator()
    generator.train(melody_corpus)
    print("Training complete.")

    print("\nGenerating and playing 10 new, unique sounds...")
    for i in range(10):
        print(f"\n--- Sound #{i+1} ---")
        
        new_melody = generator.generate(length=random.randint(6, 12))
        print(f"Generated melody: {new_melody}")

        params = {
            "melody_contour": new_melody,
            "rhythm_pattern": [random.choice([0.25, 0.5, 1.0])],
            "tempo_bpm": random.randint(90, 180),
            "base_freq": random.choice([261.63, 329.63, 440, 523.25]),
            "waveform": random.choice(["sine", "square", "saw", "triangle"]),
            "env": dict(
                attack=random.uniform(0.01, 0.2),
                decay=random.uniform(0.1, 0.4),
                sustain=random.uniform(0.4, 0.8),
                release=random.uniform(0.2, 0.6)
            )
        }
        
        # c. Generate and play the sound
        sound = generate_alarm(**params)
        sd.play(sound, 44100)
        sd.wait()
        time.sleep(0.5)
        
    print("\nFinished generating sounds.")