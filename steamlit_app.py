import streamlit as st
from pathlib import Path
import os
import json
import numpy as np
from utils import (
    get_face_encoding_from_image,
    get_voice_embedding_from_wav,
    save_user_embeddings,
    load_user_embeddings,
    list_users,
    is_face_match,
    is_voice_match,
)

BASE_USERS_DIR = "users"

st.set_page_config(page_title="Biometric Security (Face + Voice)", layout="centered")

st.title("🔐 Biometric Security — Face + Voice (Streamlit)")

# Sidebar navigation
page = st.sidebar.selectbox("Choose action", ["About", "Authenticate", "Admin Login / Enroll", "List Users"])

# Small helper for messages
def show_match_results(face_score, face_ok, voice_score, voice_ok):
    st.write("**Face match**: {:.3f} → {}".format(face_score, "PASS" if face_ok else "FAIL"))
    st.write("**Voice match**: {:.3f} → {}".format(voice_score, "PASS" if voice_ok else "FAIL"))
    if face_ok and voice_ok:
        st.success("✅ Authentication successful — Access Granted")
    else:
        st.error("❌ Authentication failed — Access Denied")

# -------- ABOUT PAGE --------
if page == "About":
    st.markdown("""
    **System overview**
    - Enroll users with a clear face image and a WAV voice sample.
    - Admin gated enrollment: after logging in as an existing user (biometric login), you can enroll new users.
    - Authentication requires both face + voice to match stored embeddings.
    
    **Notes**
    - The app stores embeddings (face_enc.npy, voice_enc.npy) per user — not raw files.
    - Tweak thresholds in the code for your environment.
    """)
    st.markdown("**Available users:** " + ", ".join(list_users(BASE_USERS_DIR)))

# -------- AUTHENTICATE PAGE --------
elif page == "Authenticate":
    st.header("Authenticate — Upload your face image + voice WAV")
    st.info("You can use `st.camera_input` (if available) or upload an image and a WAV audio file.")

    # Image input
    img_input = None
    if "camera_input" in dir(st):
        img_input = st.camera_input("Capture photo (camera) — optional")
    if not img_input:
        img_input = st.file_uploader("Upload face image (jpg/png)", type=["jpg", "jpeg", "png"])

    # Audio input
    audio_file = st.file_uploader("Upload voice sample (WAV preferred)", type=["wav", "mp3", "ogg"])

    target_user = st.text_input("Username to verify against (exact username)")
    if st.button("Verify"):

        if not target_user:
            st.error("Enter the username to verify against.")
        elif not (audio_file and img_input):
            st.error("Please upload both face image and voice sample.")
        else:
            try:
                # face enc
                # For camera_input, we get an UploadedFile-like object; pass .getvalue()
                img_bytes = img_input.getvalue() if hasattr(img_input, "getvalue") else img_input
                face_enc_live = get_face_encoding_from_image(img_bytes)

                # voice enc
                wav_bytes = audio_file.getvalue()
                voice_enc_live = get_voice_embedding_from_wav(wav_bytes)

                # load stored
                face_enc_stored, voice_enc_stored, meta = load_user_embeddings(BASE_USERS_DIR, target_user)

                face_ok, face_score = is_face_match(face_enc_live, face_enc_stored)
                voice_ok, voice_score = is_voice_match(voice_enc_live, voice_enc_stored)

                show_match_results(face_score, face_ok, voice_score, voice_ok)

            except Exception as e:
                st.exception(e)

# -------- ADMIN LOGIN / ENROLL PAGE --------
elif page == "Admin Login / Enroll":
    st.header("Admin Login (biometric) — then enroll new users")

    with st.expander("Step 1: Admin biometric login (existing user)"):
        st.info("Log in with an existing user to access enrollment panel. If no users exist yet, use the 'Initialize Admin' form below.")
        admin_user = st.text_input("Admin username (existing)")
        admin_img = st.file_uploader("Upload admin face image (jpg/png)", key="admin_img")
        admin_audio = st.file_uploader("Upload admin voice (wav)", key="admin_audio")

        admin_authenticated = False
        if st.button("Admin Verify"):
            if not (admin_user and admin_img and admin_audio):
                st.error("Provide admin username, image, and audio.")
            else:
                try:
                    face_enc_live = get_face_encoding_from_image(admin_img.getvalue())
                    voice_enc_live = get_voice_embedding_from_wav(admin_audio.getvalue())
                    face_enc_stored, voice_enc_stored, _ = load_user_embeddings(BASE_USERS_DIR, admin_user)
                    face_ok, face_score = is_face_match(face_enc_live, face_enc_stored)
                    voice_ok, voice_score = is_voice_match(voice_enc_live, voice_enc_stored)
                    if face_ok and voice_ok:
                        st.success("Admin authenticated ✅. You can enroll new users below.")
                        admin_authenticated = True
                    else:
                        st.error("Admin biometric authentication failed.")
                except Exception as e:
                    st.exception(e)

    with st.expander("Step 0: Initialize first admin (if no users exist)"):
        if len(list_users(BASE_USERS_DIR)) == 0:
            st.warning("No users found. Initialize first admin here.")
            init_user = st.text_input("New Admin Username (first admin)")
            init_face = st.file_uploader("Face image for admin (jpg/png)", key="init_face")
            init_voice = st.file_uploader("Voice WAV for admin (wav)", key="init_voice")
            if st.button("Create Admin"):
                if not (init_user and init_face and init_voice):
                    st.error("Provide username, face image and voice WAV.")
                else:
                    try:
                        fenc = get_face_encoding_from_image(init_face.getvalue())
                        venc = get_voice_embedding_from_wav(init_voice.getvalue())
                        save_user_embeddings(BASE_USERS_DIR, init_user, fenc, venc, meta={"role":"admin"})
                        st.success(f"Admin {init_user} initialized. You can now login using Admin Verify.")
                    except Exception as e:
                        st.exception(e)
        else:
            st.info("Users already exist; initializing admin not required.")

    # Enrollment panel (visible after a successful admin verification)
    st.markdown("---")
    st.subheader("Enroll new user (Admin only)")

    # We will rely on a quick manual flag: admin_authenticated is local; for a real app you'd manage session state.
    # For user convenience here: ask admin to tick a checkbox after successful login
    admin_flag = st.checkbox("I confirm I successfully authenticated as Admin", key="admin_flag")
    if admin_flag:
        new_username = st.text_input("New user username (one word, no spaces)")
        new_face = st.file_uploader("New user face image (jpg/png)", key="new_face")
        new_voice = st.file_uploader("New user voice WAV (wav)", key="new_voice")
        if st.button("Enroll New User"):
            if not (new_username and new_face and new_voice):
                st.error("Provide username, image, and voice WAV.")
            else:
                if new_username in list_users(BASE_USERS_DIR):
                    st.error("Username already exists.")
                else:
                    try:
                        fenc = get_face_encoding_from_image(new_face.getvalue())
                        venc = get_voice_embedding_from_wav(new_voice.getvalue())
                        save_user_embeddings(BASE_USERS_DIR, new_username, fenc, venc, meta={"created_by":"admin"})
                        st.success(f"User '{new_username}' enrolled successfully.")
                    except Exception as e:
                        st.exception(e)

# -------- LIST USERS PAGE --------
elif page == "List Users":
    st.header("Registered Users")
    users = list_users(BASE_USERS_DIR)
    if len(users) == 0:
        st.info("No users enrolled yet.")
    else:
        st.write(users)
        if st.button("Show user details (meta)"):
            for u in users:
                try:
                    _, _, meta = load_user_embeddings(BASE_USERS_DIR, u)
                    st.write(u, meta)
                except Exception as e:
                    st.write(u, "error reading meta:", e)
