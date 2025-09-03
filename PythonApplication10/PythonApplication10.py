import librosa
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.signal import find_peaks
from uuid import uuid4
import seaborn as sns
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine

def select_audio_file(title="Select Audio File"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")]
    )
    root.destroy()
    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    print("No file selected")
    return None

def validate_fingerprint(fingerprint, min_peaks=100, max_duration=600):
    try:
        if len(fingerprint['peaks']) < min_peaks:
            return False, f"Too few peaks ({len(fingerprint['peaks'])} < {min_peaks})"
        if fingerprint['tempo'] <= 0 or fingerprint['tempo'] > 300:
            return False, f"Invalid tempo: {fingerprint['tempo']} BPM"
        if fingerprint['duration'] <= 0 or fingerprint['duration'] > max_duration:
            return False, f"Invalid duration: {fingerprint['duration']} seconds"
        if fingerprint['mfcc_mean'].shape != (13,):
            return False, f"Invalid MFCC mean shape: {fingerprint['mfcc_mean'].shape}"
        if fingerprint['chroma_mean'].shape != (12,):
            return False, f"Invalid chroma mean shape: {fingerprint['chroma_mean'].shape}"
        if np.any(np.isnan(fingerprint['mfcc_mean'])) or np.any(np.isnan(fingerprint['chroma_mean'])):
            return False, "NaN values found in MFCC or chroma mean"
        time_frames = fingerprint['peaks']['time_frame']
        if len(time_frames) > 0:
            time_span = max(time_frames) - min(time_frames)
            if time_span < fingerprint['duration'] * 0.5:
                return False, "Peaks are too concentrated in time"
        return True, "Fingerprint is valid"
    except KeyError as e:
        return False, f"Missing key in fingerprint: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_audio_fingerprint(file_path, output_dir="fingerprints", sr=22050, n_fft=2048, hop_length=512, peak_height=-40, peak_distance=10, peak_prominence=3):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        y, sr = librosa.load(file_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1:
            raise ValueError("Audio file is too short")
        
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)
        
        peaks = []
        for t in range(stft_db.shape[1]):
            frame_peaks, properties = find_peaks(stft_db[:, t], height=peak_height, distance=peak_distance, prominence=peak_prominence)
            for f_idx, peak_idx in enumerate(frame_peaks):
                freq = peak_idx * sr / n_fft
                if 200 < freq < 5000:
                    amplitude = properties['peak_heights'][f_idx]
                    peaks.append((peak_idx, t, amplitude))
        
        peaks = np.array(peaks, dtype=[('freq_bin', int), ('time_frame', int), ('amplitude', float)])
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_mean = mfcc.mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = chroma.mean(axis=1)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        
        fingerprint = {
            'peaks': peaks,
            'mfcc_mean': mfcc_mean,
            'chroma_mean': chroma_mean,
            'tempo': tempo_value,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr,
            'duration': duration,
            'file_path': file_path
        }
        
        is_valid, validation_message = validate_fingerprint(fingerprint)
        if not is_valid:
            raise ValueError(f"Invalid fingerprint: {validation_message}")
        
        fingerprint_id = str(uuid4())
        fingerprint_path = os.path.join(output_dir, f"fingerprint_{fingerprint_id}.pkl")
        with open(fingerprint_path, 'wb') as f:
            pickle.dump(fingerprint, f)
        
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
        plt.scatter(peaks['time_frame'] * hop_length / sr, peaks['freq_bin'] * sr / n_fft, c='red', s=10, label='Peaks')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend(loc='upper right')
        spectrogram_path = os.path.join(output_dir, f"spectrogram_{fingerprint_id}.png")
        plt.savefig(spectrogram_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', hop_length=hop_length)
        plt.colorbar()
        plt.title(f'Chromagram - {os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Class')
        chromagram_path = os.path.join(output_dir, f"chromagram_{fingerprint_id}.png")
        plt.savefig(chromagram_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(mfcc, x_axis='time', hop_length=hop_length)
        plt.colorbar()
        plt.title(f'MFCC - {os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        mfcc_path = os.path.join(output_dir, f"mfcc_{fingerprint_id}.png")
        plt.savefig(mfcc_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        for path in [spectrogram_path, chromagram_path, mfcc_path]:
            if not os.path.exists(path) or os.path.getsize(path) < 1000:
                raise RuntimeError(f"Visualization file is missing or too small: {path}")
        
        return {
            'fingerprint_path': fingerprint_path,
            'spectrogram_path': spectrogram_path,
            'chromagram_path': chromagram_path,
            'mfcc_path': mfcc_path,
            'num_peaks': len(peaks),
            'tempo': tempo_value,
            'duration': duration,
            'validation_message': validation_message
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def compare_fingerprints(pkl_path1, pkl_path2, output_dir="comparisons", sr=22050, hop_length=512, n_fft=2048):
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(pkl_path1, 'rb') as f1, open(pkl_path2, 'rb') as f2:
            fp1 = pickle.load(f1)
            fp2 = pickle.load(f2)
        
        # Normalize time by duration and frequency for peaks
        peaks1 = np.array([(p['time_frame'] * hop_length / sr / fp1['duration'], 
                            p['freq_bin'] * sr / n_fft / 5000) for p in fp1['peaks']])
        peaks2 = np.array([(p['time_frame'] * hop_length / sr / fp2['duration'], 
                            p['freq_bin'] * sr / n_fft / 5000) for p in fp2['peaks']])
        peak_similarity = 0
        if len(peaks1) > 0 and len(peaks2) > 0:
            peaks1 = peaks1[:min(10000, len(peaks1))]
            peaks2 = peaks2[:min(10000, len(peaks2))]
            if peaks1.size > 0 and peaks2.size > 0:
                distance, _ = fastdtw(peaks1, peaks2, dist=euclidean)
                # Calculate max distance based on peak count and variance
                time_std1 = np.std(peaks1[:, 0]) if len(peaks1) > 1 else 1.0
                time_std2 = np.std(peaks2[:, 0]) if len(peaks2) > 1 else 1.0
                freq_std1 = np.std(peaks1[:, 1]) if len(peaks1) > 1 else 1.0
                freq_std2 = np.std(peaks2[:, 1]) if len(peaks2) > 1 else 1.0
                scale_factor = max(len(peaks1), len(peaks2)) / 2
                max_distance = np.sqrt((time_std1 + time_std2)**2 + (freq_std1 + freq_std2)**2) * scale_factor * 2  # Extra scaling
                peak_similarity = max(0, min(100, (1 - distance / max_distance) * 100 * 1.2))  # Amplify sensitivity
                print(f"Debug: Peak distance = {distance}, Max distance = {max_distance}, Scale factor = {scale_factor}")
        
        mfcc1, mfcc2 = fp1['mfcc_mean'].flatten(), fp2['mfcc_mean'].flatten()
        mfcc_distance = euclidean(mfcc1, mfcc2)
        mfcc_similarity = max(0, min(100, (1 - mfcc_distance / (np.std(mfcc1) + np.std(mfcc2) + 1e-6)) * 100))
        
        chroma1, chroma2 = fp1['chroma_mean'].flatten(), fp2['chroma_mean'].flatten()
        chroma1 = (chroma1 - np.min(chroma1)) / (np.max(chroma1) - np.min(chroma1) + 1e-6)
        chroma2 = (chroma2 - np.min(chroma2)) / (np.max(chroma2) - np.min(chroma2) + 1e-6)
        chroma_cosine = cosine(chroma1, chroma2)
        chroma_similarity = max(0, min(100, (1 - chroma_cosine / 2) * 100))
        print(f"Debug: Chroma cosine distance = {chroma_cosine}")
        
        tempo_diff = abs(fp1['tempo'] - fp2['tempo'])
        tempo_similarity = max(0, 1 - tempo_diff / max(fp1['tempo'], fp2['tempo'])) * 100
        
        centroid_diff = abs(fp1['spectral_centroid'].mean() - fp2['spectral_centroid'].mean())
        centroid_similarity = max(0, 1 - centroid_diff / max(fp1['spectral_centroid'].mean(), fp2['spectral_centroid'].mean())) * 100
        
        rolloff_diff = abs(fp1['spectral_rolloff'].mean() - fp2['spectral_rolloff'].mean())
        rolloff_similarity = max(0, 1 - rolloff_diff / max(fp1['spectral_rolloff'].mean(), fp2['spectral_rolloff'].mean())) * 100
        
        zcr_diff = abs(fp1['zcr'].mean() - fp2['zcr'].mean())
        zcr_similarity = max(0, 1 - zcr_diff / max(fp1['zcr'].mean(), fp2['zcr'].mean())) * 100
        
        weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
        overall_similarity = (
            weights[0] * peak_similarity + weights[1] * mfcc_similarity +
            weights[2] * chroma_similarity + weights[3] * tempo_similarity +
            weights[4] * centroid_similarity + weights[5] * rolloff_similarity +
            weights[6] * zcr_similarity
        )
        overall_similarity = max(0, min(100, overall_similarity))
        
        # Visualization: Bar plot for similarity metrics
        plt.style.use('seaborn-v0_8')
        metrics = ['Overall', 'Peaks', 'MFCC', 'Chroma', 'Tempo', 'Centroid', 'Rolloff', 'ZCR']
        values = [overall_similarity, peak_similarity, mfcc_similarity, chroma_similarity,
                  tempo_similarity, centroid_similarity, rolloff_similarity, zcr_similarity]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, values, color='skyblue', edgecolor='black')
        plt.ylim(0, 100)
        plt.title(f'Similarity: {os.path.basename(fp1["file_path"])} vs {os.path.basename(fp2["file_path"])}')
        plt.ylabel('Similarity (%)')
        plt.xlabel('Metrics')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        similarity_plot = os.path.join(output_dir, f"similarity_{uuid4()}.png")
        plt.savefig(similarity_plot, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Visualization: MFCC and Chroma mean comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(mfcc1, label=os.path.basename(fp1['file_path']), color='blue')
        plt.plot(mfcc2, label=os.path.basename(fp2['file_path']), color='orange')
        plt.title('MFCC Mean Comparison')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(chroma1, label=os.path.basename(fp1['file_path']), color='blue')
        plt.plot(chroma2, label=os.path.basename(fp2['file_path']), color='orange')
        plt.title('Chroma Mean Comparison')
        plt.xlabel('Pitch Class Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        feature_plot = os.path.join(output_dir, f"feature_comparison_{uuid4()}.png")
        plt.savefig(feature_plot, bbox_inches='tight', dpi=150)
        plt.close()
        
        summary = []
        if overall_similarity > 80:
            summary.append("Songs are highly similar.")
        elif overall_similarity > 50:
            summary.append("Songs have moderate similarity.")
        else:
            summary.append("Songs are quite different.")
        if peak_similarity < 50:
            summary.append("Notable differences in peak distribution (frequency-time patterns).")
        if mfcc_similarity < 50:
            summary.append("Timbral characteristics (MFCC) differ notably.")
        if chroma_similarity < 50:
            summary.append("Harmonic content (chroma) shows some differences.")
        if tempo_similarity < 50:
            summary.append("Tempos are significantly different.")
        
        return {
            'overall_similarity': overall_similarity,
            'peak_similarity': peak_similarity,
            'mfcc_similarity': mfcc_similarity,
            'chroma_similarity': chroma_similarity,
            'tempo_similarity': tempo_similarity,
            'centroid_similarity': centroid_similarity,
            'rolloff_similarity': rolloff_similarity,
            'zcr_similarity': zcr_similarity,
            'differences': [
                f"Peaks: {len(fp1['peaks'])} vs {len(fp2['peaks'])}",
                f"MFCC mean: {fp1['mfcc_mean'].mean():.2f} vs {fp2['mfcc_mean'].mean():.2f}",
                f"Chroma mean: {fp1['chroma_mean'].mean():.2f} vs {fp2['chroma_mean'].mean():.2f}",
                f"Tempo: {fp1['tempo']:.2f} BPM vs {fp2['tempo']:.2f} BPM",
                f"Spectral Centroid: {fp1['spectral_centroid'].mean():.2f} Hz vs {fp2['spectral_centroid'].mean():.2f} Hz",
                f"Spectral Rolloff: {fp1['spectral_rolloff'].mean():.2f} Hz vs {fp2['spectral_rolloff'].mean():.2f} Hz",
                f"Zero Crossing Rate: {fp1['zcr'].mean():.4f} vs {fp2['zcr'].mean():.4f}"
            ],
            'summary': summary,
            'similarity_plot': similarity_plot,
            'feature_plot': feature_plot
        }
    
    except Exception as e:
        print(f"Error comparing fingerprints: {str(e)}")
        return None






def inspect_fingerprint(pkl_path, hop_length=512, sr=22050, n_fft=2048):
    try:
        with open(pkl_path, 'rb') as f:
            fingerprint = pickle.load(f)
        
        is_valid, message = validate_fingerprint(fingerprint)
        print(f"Validation: {message}")
        print("Fingerprint Contents:")
        print(f"- File: {fingerprint['file_path']}")
        print(f"- Number of peaks: {len(fingerprint['peaks'])}")
        print(f"- Tempo: {fingerprint['tempo']} BPM")
        print(f"- Duration: {fingerprint['duration']} seconds")
        print(f"- MFCC mean shape: {fingerprint['mfcc_mean'].shape}")
        print(f"- Chroma mean shape: {fingerprint['chroma_mean'].shape}")
        print(f"- Spectral Centroid shape: {fingerprint['spectral_centroid'].shape}")
        print(f"- Spectral Rolloff shape: {fingerprint['spectral_rolloff'].shape}")
        print(f"- Zero Crossing Rate shape: {fingerprint['zcr'].shape}")
        print(f"- First 5 peaks (freq_bin, time_frame, amplitude):")
        for peak in fingerprint['peaks'][:5]:
            print(f"  {peak}")
        
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(10, 6))
        plt.scatter(fingerprint['peaks']['time_frame'] * hop_length / sr,
                    fingerprint['peaks']['freq_bin'] * sr / n_fft,
                    c='blue', s=10, alpha=0.5)
        plt.title(f'Peak Distribution - {os.path.basename(fingerprint["file_path"])}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Error inspecting fingerprint: {str(e)}")

if __name__ == "__main__":
    print("Select the first audio file:")
    audio_file1 = select_audio_file(title="Select First Audio File")
    if not audio_file1:
        audio_file1 = "Vanished - Crystal Castles.mp3"
        print(f"Using default file for first song: {audio_file1}")
    
    result1 = create_audio_fingerprint(audio_file1)
    if result1:
        print(f"Fingerprint saved at: {result1['fingerprint_path']}")
        print(f"Spectrogram saved at: {result1['spectrogram_path']}")
        print(f"Chromagram saved at: {result1['chromagram_path']}")
        print(f"MFCC saved at: {result1['mfcc_path']}")
        print(f"Number of peaks: {result1['num_peaks']}")
        print(f"Tempo: {result1['tempo']} BPM")
        print(f"Duration: {result1['duration']} seconds")
        print(f"Validation: {result1['validation_message']}")
        inspect_fingerprint(result1['fingerprint_path'])
    
    print("\nSelect the second audio file for comparison:")
    audio_file2 = select_audio_file(title="Select Second Audio File")
    if audio_file2:
        result2 = create_audio_fingerprint(audio_file2)
        if result2:
            print(f"Fingerprint saved at: {result2['fingerprint_path']}")
            print(f"Spectrogram saved at: {result2['spectrogram_path']}")
            print(f"Chromagram saved at: {result2['chromagram_path']}")
            print(f"MFCC saved at: {result2['mfcc_path']}")
            print(f"Number of peaks: {result2['num_peaks']}")
            print(f"Tempo: {result2['tempo']} BPM")
            print(f"Duration: {result2['duration']} seconds")
            print(f"Validation: {result2['validation_message']}")
            inspect_fingerprint(result2['fingerprint_path'])
            
            comparison = compare_fingerprints(result1['fingerprint_path'], result2['fingerprint_path'])
            if comparison:
                print("\nComparison Results:")
                print(f"Overall Similarity: {comparison['overall_similarity']:.2f}%")
                print(f"Peak Similarity: {comparison['peak_similarity']:.2f}%")
                print(f"MFCC Similarity: {comparison['mfcc_similarity']:.2f}%")
                print(f"Chroma Similarity: {comparison['chroma_similarity']:.2f}%")
                print(f"Tempo Similarity: {comparison['tempo_similarity']:.2f}%")
                print(f"Spectral Centroid Similarity: {comparison['centroid_similarity']:.2f}%")
                print(f"Spectral Rolloff Similarity: {comparison['rolloff_similarity']:.2f}%")
                print(f"Zero Crossing Rate Similarity: {comparison['zcr_similarity']:.2f}%")
                print("\nDifferences:")
                for diff in comparison['differences']:
                    print(f"  - {diff}")
                print("\nSummary:")
                for summary_item in comparison['summary']:
                    print(f"  - {summary_item}")
                print(f"\nSimilarity plot saved at: {comparison['similarity_plot']}")
                print(f"Feature comparison plot saved at: {comparison['feature_plot']}")
                for path in [comparison['similarity_plot'], comparison['feature_plot']]:
                    img = Image.open(path)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(os.path.basename(path))
                    plt.show()
    else:
        print("No second audio file selected. Skipping comparison.")