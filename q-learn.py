import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import random
import threading
import time
import matplotlib.pyplot as plt

def generate_tone(frequency, duration, sample_rate=44100, waveform="sine"):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if waveform == "sine": tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    elif waveform == "square": tone = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "triangle": tone = 0.5 * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
    elif waveform == "saw": tone = 0.5 * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
    else: tone = np.random.normal(0, 0.3, len(t))
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
        if semitone_shift is None: continue
        # For this version, rhythm pattern is now part of the preset
        duration = rhythm_pattern[i % len(rhythm_pattern)] * beat_sec
        freq = base_freq * (2 ** (semitone_shift / 12))
        tone = generate_tone(freq, duration, waveform=waveform, sample_rate=sample_rate)
        tone = apply_envelope(tone, **env, sample_rate=sample_rate)
        full_signal = np.concatenate([full_signal, tone])
    max_amp = np.max(np.abs(full_signal))
    if max_amp > 0: full_signal /= max_amp
    return full_signal

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, exploration_rate=1.0, exploration_decay=0.998):
        self.actions = actions
        self.alpha = learning_rate
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = np.zeros(len(actions))
    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon: return random.randint(0, len(self.actions) - 1)
        else: return np.argmax(self.q_table)
    def update_q_table(self, action, reward):
        self.q_table[action] = self.q_table[action] * (1 - self.alpha) + self.alpha * reward
        if self.epsilon > 0.01: self.epsilon *= self.epsilon_decay


class TrainingLogger:
    def __init__(self):
        self.trials, self.rewards, self.epsilons, self.best_q_values = [], [], [], []
    def log(self, trial, reward, epsilon, q_table):
        self.trials.append(trial)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.best_q_values.append(np.max(q_table))
    def plot_all(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1); plt.plot(self.trials, self.rewards, color="black", label="Reward per Trial"); plt.title("Reward Progress"); plt.grid(True); plt.legend()
        plt.subplot(3, 1, 2); plt.plot(self.trials, self.epsilons, color="black", label="Exploration Rate (Epsilon)"); plt.title("Exploration Decay"); plt.grid(True); plt.legend()
        plt.subplot(3, 1, 3); plt.plot(self.trials, self.best_q_values, color="black", label="Best Q-Value"); plt.title("Q-Value Growth"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig("training_progress.png", dpi=200); plt.show()
        print("✅ Saved plot as 'training_progress.png'")

class AlarmApp:
    def __init__(self, root, agent, logger):
        self.root = root
        self.agent = agent
        self.logger = logger
        self.root.title("Q-Learning Alarm (Curated Sounds)")
        self.root.geometry("450x600")

        self.status_label = ttk.Label(root, text="Click 'Start' to find your best alarm.", font=("Helvetica", 12), justify=tk.CENTER)
        self.status_label.pack(pady=20)
        self.wake_button = ttk.Button(root, text="I'M AWAKE! (Give Reward)", command=self.give_reward, state=tk.DISABLED)
        self.wake_button.pack(pady=10, ipady=10)
        self.start_button = ttk.Button(root, text="Start", command=self.start_simulation)
        self.start_button.pack(pady=5)
        
        self.q_table_frame = ttk.LabelFrame(root, text="Agent's Beliefs (Alarm Effectiveness)")
        self.q_table_frame.pack(pady=10, padx=10, fill="x")
        self.q_labels = []
        for i, action in enumerate(self.agent.actions):
            label = ttk.Label(self.q_table_frame, text=f"{action['name']}: {self.agent.q_table[i]:.2f}")
            label.pack(anchor="w", padx=5)
            self.q_labels.append(label)

        ttk.Button(root, text="Show Training Plot", command=self.logger.plot_all).pack(pady=10)

        self.trial_count = 0
        self.last_action_index = None
        self.timeout_job = None
        self.start_time = None

    def start_simulation(self):
        self.start_button.config(state=tk.DISABLED)
        self.run_next_trial()

    def run_next_trial(self):
        self.trial_count += 1
        sd.stop()
        self.wake_button.config(state=tk.DISABLED)
        
        # Agent chooses a pre-defined sound
        self.last_action_index = self.agent.choose_action()
        action_details = self.agent.actions[self.last_action_index]
        
        self.status_label.config(text=f"Trial {self.trial_count}: Agent chose '{action_details['name']}'\nEpsilon: {self.agent.epsilon:.2f}")
        
        # Generate the sound directly from the action's parameters
        sound = generate_alarm(**action_details['params'])
        threading.Thread(target=sd.play, args=(sound, 44100), daemon=True).start()
        
        self.wake_button.config(state=tk.NORMAL)
        self.start_time = time.time()
        self.timeout_job = self.root.after(10000, self.handle_timeout)
        self.update_q_display()

    def give_reward(self):
        if self.timeout_job: self.root.after_cancel(self.timeout_job)
        response_time = time.time() - self.start_time
        reward = max(0.5, 15 - response_time)
        self.agent.update_q_table(self.last_action_index, reward)
        self.logger.log(self.trial_count, reward, self.agent.epsilon, self.agent.q_table)
        self.status_label.config(text=f"REWARD! ({reward:.2f}) for '{self.agent.actions[self.last_action_index]['name']}'")
        self.root.after(1500, self.run_next_trial)

    def handle_timeout(self):
        reward = -5
        self.agent.update_q_table(self.last_action_index, reward)
        self.logger.log(self.trial_count, reward, self.agent.epsilon, self.agent.q_table)
        self.status_label.config(text=f"TIMEOUT! Negative reward ({reward}) given.")
        self.root.after(1500, self.run_next_trial)

    def update_q_display(self):
        for i, label in enumerate(self.q_labels):
            label.config(text=f"{self.agent.actions[i]['name']}: {self.agent.q_table[i]:.2f}")

if __name__ == "__main__":
    sound_presets = {
        "Major Scale": {"melody_contour": [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0], "rhythm_pattern": [0.5], "tempo_bpm": 140, "base_freq": 523.25, "env": dict(attack=0.01, decay=0.2, sustain=0.7, release=0.2), "waveform": "sine"},
        "Minor Lament": {"melody_contour": [0, 2, 3, 5, 7, 8, 10, 12, 10, 8, 7, 5, 3, 2, 0], "rhythm_pattern": [0.75], "tempo_bpm": 90, "base_freq": 440.0, "env": dict(attack=0.4, decay=0.5, sustain=0.5, release=0.8), "waveform": "triangle"},
        "Pentatonic Riff": {"melody_contour": [0, 2, 4, 7, 9, 12, 9, 7, 4, 2, 0], "rhythm_pattern": [0.25], "tempo_bpm": 180, "base_freq": 329.63, "env": dict(attack=0.01, decay=0.3, sustain=0.1, release=0.2), "waveform": "square"},
        "Sci-Fi Warning": {"melody_contour": [0, 7, 0, 7, 0, 7, 0, 7], "rhythm_pattern": [0.5], "tempo_bpm": 160, "base_freq": 880.0, "env": dict(attack=0.01, decay=0.1, sustain=0.1, release=0.1), "waveform": "saw"},
        "Retro Power-Up": {"melody_contour": [0, 4, 7, 12], "rhythm_pattern": [0.2], "tempo_bpm": 200, "base_freq": 261.63, "env": dict(attack=0.005, decay=0.1, sustain=0.6, release=0.1), "waveform": "square"},
        "Game Over Fall": {"melody_contour": [12, 8, 5, 0, -4, -7, -12], "rhythm_pattern": [0.5], "tempo_bpm": 100, "base_freq": 440.0, "env": dict(attack=0.01, decay=0.3, sustain=0.3, release=0.5), "waveform": "square"},
        "Tension Riser": {"melody_contour": list(range(13)), "rhythm_pattern": [0.25], "tempo_bpm": 120, "base_freq": 220.0, "env": dict(attack=0.5, decay=0, sustain=1.0, release=0.1), "waveform": "saw"},
        "Busy Bee Trill": {"melody_contour": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], "rhythm_pattern": [0.15], "tempo_bpm": 220, "base_freq": 659.25, "env": dict(attack=0.005, decay=0.1, sustain=0.8, release=0.1), "waveform": "saw"},
        "Emergency Broadcast": {"melody_contour": [12, 12, 12, 7, 7, 7, 12, 12, 12], "rhythm_pattern": [0.5, 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5, 1], "tempo_bpm": 150, "base_freq": 523.25, "env": dict(attack=0.01, decay=0.1, sustain=0.9, release=0.2), "waveform": "square"},
    }

    ALARM_ACTIONS = []
    for name, params in sound_presets.items():
        ALARM_ACTIONS.append({"name": name, "params": params})
    
    q_agent = QLearningAgent(actions=ALARM_ACTIONS)
    logger = TrainingLogger()
    main_window = tk.Tk()
    app = AlarmApp(main_window, q_agent, logger)
    main_window.mainloop()