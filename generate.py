import numpy as np
import sounddevice as sd
import time
import random
from scipy.io import wavfile # Import the wavfile module

# --- Advanced Synthesis & FX Engine ---
# Engine is tailored for creating natural, atmospheric sounds.

def generate_tone(frequency, duration, sample_rate=44100, waveform="sine"):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if waveform == "sine": return np.sin(2 * np.pi * frequency * t)
    if waveform == "triangle": return (2/np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
    return np.sin(2 * np.pi * frequency * t)

def apply_stereo_pan(signal, pan):
    """Pans a mono signal to a stereo position (-1 Left, 0 Center, 1 Right)."""
    # If pan is an array, ensure signal can be broadcast correctly
    if isinstance(pan, np.ndarray):
        signal = signal[:, np.newaxis]
        pan = pan[:, np.newaxis]

    pan = np.clip(pan, -1, 1)
    left_gain = np.cos((pan + 1) * np.pi / 4)
    right_gain = np.sin((pan + 1) * np.pi / 4)
    
    # Handle both scalar and array pans
    if signal.ndim == 1:
        return np.vstack((signal * left_gain, signal * right_gain)).T
    else: # signal is already (N, 1)
        return np.hstack((signal * left_gain, signal * right_gain))


def apply_envelope(signal, attack, decay, sustain_level, release):
    """Applies a standard ADSR envelope."""
    n = len(signal)
    sample_rate=44100
    a, d, r = max(1, int(attack*sample_rate)), max(1, int(decay*sample_rate)), max(1, int(release*sample_rate))
    s = max(0, n - (a + d + r))
    envelope = np.concatenate([np.linspace(0, 1, a)**2, np.linspace(1, sustain_level, d),
                               np.ones(s) * sustain_level, np.linspace(sustain_level, 0, r)**2])
    if len(envelope) < n: envelope = np.pad(envelope, (0, n - len(envelope)), 'constant')
    else: envelope = envelope[:n]
    return signal * envelope

def apply_reverb(signal, delay_times, decays, sample_rate=44100):
    """Simulates a rich reverb by combining multiple echoes."""
    output = np.copy(signal)
    for delay_sec, decay in zip(delay_times, decays):
        delay_samples = int(delay_sec * sample_rate)
        delayed_signal = np.zeros_like(signal)
        if delay_samples < len(signal):
            delayed_signal[delay_samples:] = signal[:-delay_samples]
        output += delayed_signal * decay
    return output / (1 + sum(decays))


# --- The 5 Chill, Engaging, Nature-Inspired Alarms ---

def create_alarm_1_crystal_cave_drops():
    """
    DESIGN: Simulates water droplets echoing in a large, resonant cave. The randomness of
    the timing, pitch, and stereo position is highly engaging but gentle. The heavy reverb
    creates a deeply calming sense of space.
    """
    duration = 15.0
    sample_rate = 44100
    final_signal = np.zeros(int(duration * sample_rate)) # Start as mono
    
    scale = [0, 4, 7, 12, 16]
    
    for _ in range(40):
        drop_time = random.uniform(0, duration - 1)
        drop_pitch = 523.25 * (2**(random.choice(scale) / 12)) * random.uniform(0.99, 1.01)
        
        drop_signal = generate_tone(drop_pitch, 0.5, waveform="sine")
        drop_signal = apply_envelope(drop_signal, 0.005, 0.2, 0.0, 0.2) * random.uniform(0.4, 0.8)
        
        start_sample = int(drop_time * sample_rate)
        end_sample = start_sample + len(drop_signal)
        if end_sample < len(final_signal):
            final_signal[start_sample:end_sample] += drop_signal
            
    reverb_delays = [0.21, 0.34, 0.45, 0.52, 0.67]
    reverb_decays = [0.6, 0.5, 0.4, 0.3, 0.2]
    reverberated_signal = apply_reverb(final_signal, reverb_delays, reverb_decays)
    
    # Pan the final mono signal to stereo at the end
    return apply_stereo_pan(reverberated_signal, 0)

def create_alarm_2_bonsai_garden_chimes():
    """
    DESIGN: Mimics bamboo or metal wind chimes stirred by a gentle, unpredictable breeze.
    The notes have an extremely long, shimmering decay. The clusters of notes simulate
    the chaotic yet beautiful patterns of wind.
    """
    duration = 16.0
    sample_rate = 44100
    final_signal = np.zeros((int(duration * sample_rate), 2))
    
    scale = [0, 5, 7, 9, 12, 17]
    
    for i in range(12):
        gust_time = 1.0 + i * 1.2 + random.uniform(-0.4, 0.4)
        num_chimes_in_gust = random.randint(1, 4)
        
        for j in range(num_chimes_in_gust):
            chime_time = gust_time + j * random.uniform(0.05, 0.15)
            chime_pitch = 440 * (2**(random.choice(scale) / 12))
            chime_pan = random.uniform(-1, 1)
            
            chime_signal_1 = generate_tone(chime_pitch, 4.0, waveform="sine")
            chime_signal_2 = generate_tone(chime_pitch * 2, 4.0, waveform="triangle") * 0.3
            chime_signal = apply_envelope(chime_signal_1 + chime_signal_2, 0.001, 0.1, 0.8, 4.0) * 0.6
            
            stereo_chime = apply_stereo_pan(chime_signal, chime_pan)
            
            start_sample = int(chime_time * sample_rate)
            end_sample = start_sample + len(stereo_chime)
            if end_sample < len(final_signal):
                final_signal[start_sample:end_sample] += stereo_chime
    
    return final_signal

def create_alarm_3_summer_night_field():
    """
    DESIGN: Creates an immersive, 3D soundscape of singing insects on a warm night. A deep,
    soft drone provides the "silence" of the night, while dozens of randomized "chirps"
    create a complex, living texture that is incredibly engaging and non-threatening.
    """
    duration = 14.0
    sample_rate = 44100
    
    drone = generate_tone(60, duration, waveform="sine") * 0.3
    drone = apply_envelope(drone, 6.0, 0, 1.0, 6.0)
    final_signal = apply_stereo_pan(drone, 0)
    
    for _ in range(80):
        chirp_time = random.uniform(0, duration - 0.5)
        chirp_pitch = random.uniform(2500, 4000)
        chirp_pan = random.uniform(-0.9, 0.9)
        
        chirp = generate_tone(chirp_pitch, 0.15, waveform="sine")
        chirp = apply_envelope(chirp, 0.01, 0.05, 0.1, 0.1) * random.uniform(0.05, 0.15)
        
        stereo_chirp = apply_stereo_pan(chirp, chirp_pan)
        
        start_sample = int(chirp_time * sample_rate)
        end_sample = start_sample + len(stereo_chirp)
        if end_sample < len(final_signal):
            final_signal[start_sample:end_sample] += stereo_chirp
            
    return final_signal

def create_alarm_4_deep_ocean_echoes():
    """
    DESIGN: Evokes the slow, majestic, and deeply calming calls of whales in a vast ocean.
    The sound uses slow pitch-glides and a massive, cavernous reverb. It's sparse and
    minimalist, creating a feeling of profound peace and scale.
    """
    duration = 18.0
    sample_rate = 44100
    final_signal = np.zeros(int(duration * sample_rate)) # Start as mono
    
    calls = [(2.0, 4.0, 150, 400), (7.0, 5.0, 500, 200), (12.0, 4.0, 250, 450)]
    
    for start_time, call_duration, freq_start, freq_end in calls:
        num_samples = int(call_duration * sample_rate)
        frequencies = np.linspace(freq_start, freq_end, num_samples)
        phase = np.cumsum(2 * np.pi * frequencies / sample_rate)
        
        call_signal = np.sin(phase)
        call_signal = apply_envelope(call_signal, 1.5, 1.0, 0.7, 1.5) * 0.6
        
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(call_signal)
        if end_sample < len(final_signal):
            final_signal[start_sample:end_sample] += call_signal
            
    reverb_delays = [0.5, 0.8, 1.1]
    reverb_decays = [0.7, 0.5, 0.3]
    reverberated_signal = apply_reverb(final_signal, reverb_delays, reverb_decays)
    
    return apply_stereo_pan(reverberated_signal, 0)

def create_alarm_5_aurora_borealis():
    """
    DESIGN: A sonic representation of the northern lights. A deep, stable drone is the
    dark sky, while shimmering, slowly moving curtains of sound dance above. The effect is
    ethereal, magical, and incredibly smooth. Best with headphones.
    """
    duration = 20.0
    sample_rate = 44100
    
    drone = generate_tone(70, duration, waveform="sine") * 0.4
    drone = apply_envelope(drone, 8.0, 0, 1.0, 8.0)
    final_signal = apply_stereo_pan(drone, 0)
    
    for i in range(3):
        curtain_pitch = 440 * (2**(random.choice([7, 11, 14, 16]) / 12))
        curtain = generate_tone(curtain_pitch, duration, waveform="sine") * 0.2
        
        # LFOs for panning and volume to create movement
        pan_lfo = np.sin(2 * np.pi * random.uniform(0.05, 0.15) * np.linspace(0, duration, len(curtain)))
        vol_lfo = (np.sin(2 * np.pi * random.uniform(0.1, 0.2) * np.linspace(0, duration, len(curtain))) + 1) / 2
        
        curtain *= vol_lfo
        curtain = apply_envelope(curtain, 5.0, 0, 1.0, 5.0)
        
        stereo_curtain = apply_stereo_pan(curtain, pan_lfo)
        final_signal += stereo_curtain
        
    return final_signal

# --- Main Execution Logic ---

if __name__ == "__main__":
    alarms = {
        "1_Crystal_Cave_Drops": create_alarm_1_crystal_cave_drops,
        "2_Bonsai_Garden_Chimes": create_alarm_2_bonsai_garden_chimes,
        "3_Summer_Night_Field": create_alarm_3_summer_night_field,
        "4_Deep_Ocean_Echoes": create_alarm_4_deep_ocean_echoes,
        "5_Aurora_Borealis": create_alarm_5_aurora_borealis
    }
    
    print("Generating and playing 5 chill, engaging, and nature-inspired alarms.")
    print("Headphones are highly recommended.\n")
    
    sample_rate = 44100
    
    for name, create_func in alarms.items():
        print(f"--- Processing: {name.replace('_', ' ')} ---")
        description = ' '.join(create_func.__doc__.strip().split())
        print(f"   DESIGN: {description}")
        
        # 1. Generate the sound data
        sound = create_func()
        
        # 2. Normalize the audio to prevent clipping
        max_amp = np.max(np.abs(sound))
        if max_amp > 0:
            sound /= max_amp
            
        # 3. Save to .wav file
        # Convert float audio to 16-bit PCM for standard .wav format
        wav_data = np.int16(sound * 32767)
        filename = f"{name}.wav"
        wavfile.write(filename, sample_rate, wav_data)
        print(f"   >>> Successfully saved to '{filename}'")
        
        # 4. Play the sound
        print("   Now Playing...")
        sd.play(sound, sample_rate, blocking=True)
        time.sleep(1.0)
        
    print("\nShowcase and .wav file generation finished.")