# 🎵 Audio Fingerprinting & Similarity Search

A Python project for creating **audio fingerprints**, comparing tracks, and visualizing similarity using spectral features.  
Includes a simple **Tkinter GUI** for managing audio datasets and searching for similar songs.

---

## 📸 Preview
| Main Window | Fingerprint Matching |
|-------------|----------------------|
| ![Main](Screen/1.png) | ![Matching](Screen/2.png) |

---

## ✨ Features
- Extracts audio features with **librosa**  
- Generates **audio fingerprints** for comparison  
- Computes similarity using **FastDTW**  
- Stores fingerprints with **pickle**  
- Interactive **Tkinter interface**  
- Visualization of spectrograms & similarity  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
Create virtual environment & install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
🚀 Usage
Run the GUI:

bash
Copy code
python src/main.py
Load an audio file (.wav, .mp3)

Generate and save its fingerprint

Compare fingerprints across your dataset

Visualize spectrograms & similarity results

📦 Requirements
All dependencies are listed in requirements.txt:

shell
Copy code
numpy<2.3
librosa==0.10.1
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.2
pillow>=10.0.0
fastdtw>=0.3.4
⚠️ Note: tkinter is included with most Python distributions. If missing, install it via your package manager.

📂 Project Structure
css
Copy code
📦 your-repo
 ┣ 📂 Screen/           # screenshots
 ┣ 📂 src/              # source code
 ┃ ┗ main.py
 ┣ requirements.txt
 ┗ README.md
🛠️ Future Improvements
Support for more audio formats

Optimized fingerprint storage

Web-based interface (Flask/React)

Improved similarity metrics
