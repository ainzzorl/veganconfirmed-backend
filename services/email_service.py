import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
import logging

from services.report_service import create_html_report, save_report_locally

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self, require_email_config: bool = True):
        """
        Initialize email service with Gmail SMTP

        Args:
            require_email_config: If True, raises error when email config is missing.
                                  Set to False for local-only report generation.
        """
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_login = os.getenv("SENDER_LOGIN")  # Login username for SMTP
        self.sender_password = os.getenv("SENDER_PASSWORD")  # App password
        self.recipient_email = os.getenv("RECIPIENT_EMAIL")

        # Use sender_login if provided, otherwise fall back to sender_email
        self.smtp_username = (
            self.sender_login if self.sender_login else self.sender_email
        )

        self._email_configured = all(
            [self.sender_email, self.sender_password, self.recipient_email]
        )

        if require_email_config and not self._email_configured:
            raise ValueError(
                "Email configuration missing. Please set SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAIL environment variables."
            )

    def send_weekly_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Send weekly usage report via email

        Args:
            report_data: Dictionary containing report statistics

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self._email_configured:
            logger.error("Cannot send email: email configuration is missing")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email
            msg["Subject"] = (
                f"Vegan Confirmed Weekly Usage Report - {report_data['week_start']} to {report_data['week_end']}"
            )

            # Create HTML body
            html_body = create_html_report(report_data)
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.sender_password)
                server.send_message(msg)

            logger.info(f"Weekly report sent successfully to {self.recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")
            return False
