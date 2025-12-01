import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import face_recognition

from speechbrain.pretrained import SpeakerRecognition

_speaker_model = None
def get_speaker_model():
    global _speaker_model
    if _speaker_model is None:
        _speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
    return _speaker_model

# --------------------------------
# FACE EMBEDDING
# --------------------------------
def get_face_encoding_from_image(image_path_or_bytes):
    """
    Accepts: path to image file (str / Path) or bytes (uploaded image.getbuffer()).
    Returns: 1D numpy array face encoding or raises ValueError if no face found.
    """
    # Load image
    if isinstance(image_path_or_bytes, (str, Path)):
        img = face_recognition.load_image_file(str(image_path_or_bytes))
    else:
        # assume file-like / bytes
        import io
        img = face_recognition.load_image_file(io.BytesIO(image_path_or_bytes))

    encs = face_recognition.face_encodings(img)
    if len(encs) == 0:
        raise ValueError("No face found in the image. Please upload a clear face image.")
    # If multiple faces, take the first
    return np.array(encs[0])

# --------------------------------
# VOICE EMBEDDING
# --------------------------------
def get_voice_embedding_from_wav(wav_path_or_bytes):
    """
    Accepts: file path (str/Path) or bytes (uploaded AudioFile.getvalue()).
    Returns: 1D numpy array speaker embedding.
    Requires SpeechBrain installed and model downloaded.
    """
    model = get_speaker_model()

    # SpeechBrain expects a file path; if bytes provided, write temp file
    temp_path = None
    if isinstance(wav_path_or_bytes, (str, Path)):
        wav_path = str(wav_path_or_bytes)
    else:
        import tempfile, io
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.write(wav_path_or_bytes)
        temp.flush()
        temp.close()
        temp_path = temp.name
        wav_path = temp_path

    # encode_file returns affinity and embedding in some APIs; we use encode_file
    try:
        emb = model.encode_file(wav_path)  # torch tensor
        emb = emb.detach().cpu().numpy().squeeze()
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except:
                pass
    return emb

# --------------------------------
# MATCHING
# --------------------------------
def cosine_score(a, b):
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def is_face_match(enc1, enc2, threshold=0.55):
    # face_recognition uses Euclidean distance typically; we use cosine here. You can tune threshold.
    score = cosine_score(enc1, enc2)
    return score >= threshold, float(score)

def is_voice_match(enc1, enc2, threshold=0.6):
    score = cosine_score(enc1, enc2)
    return score >= threshold, float(score)

# --------------------------------
# STORAGE (embeddings only)
# --------------------------------
def ensure_user_dir(base_dir, username):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ud = base / username
    ud.mkdir(parents=True, exist_ok=True)
    return ud

def save_user_embeddings(base_dir, username, face_enc, voice_enc, meta=None):
    ud = ensure_user_dir(base_dir, username)
    np.save(ud / "face_enc.npy", face_enc)
    np.save(ud / "voice_enc.npy", voice_enc)
    meta_data = meta or {}
    meta_data.update({"username": username})
    with open(ud / "meta.json", "w") as f:
        json.dump(meta_data, f)

def load_user_embeddings(base_dir, username):
    ud = Path(base_dir) / username
    if not ud.exists():
        raise FileNotFoundError("User not found")
    face_enc = np.load(ud / "face_enc.npy")
    voice_enc = np.load(ud / "voice_enc.npy")
    meta = {}
    if (ud / "meta.json").exists():
        with open(ud / "meta.json", "r") as f:
            meta = json.load(f)
    return face_enc, voice_enc, meta

def list_users(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return []
    return [p.name for p in base.iterdir() if p.is_dir()]
