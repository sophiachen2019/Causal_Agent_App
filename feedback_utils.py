import smtplib
import ssl
import streamlit as st
import os

def send_email(feedback_text, user_email=None):
    """
    Sends feedback email using Gmail SMTP.
    """
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "sophiachen2012@gmail.com"  # Using user's email as sender (App Password required)
    receiver_email = "sophiachen2012@gmail.com" # Forwarding to self
    
    # Get Password from Secrets or Env
    password = None
    try:
        password = st.secrets["EMAIL_PASSWORD"]
    except (FileNotFoundError, KeyError):
        password = os.getenv("EMAIL_PASSWORD")
        
    if not password:
        return False, "Email password not configured. Please set EMAIL_PASSWORD in .streamlit/secrets.toml"

    message = f"""\
Subject: [Causal Agent App] New User Feedback

Feedback:
{feedback_text}

FROM: {user_email if user_email else 'Anonymous'}
"""

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        return True, "Feedback sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {e}"
