import functions_framework
from flask import Request, jsonify
import logging
from datetime import datetime, timedelta, timezone
from analysis_core import AnalysisCore
from utils.logger import setup_logger

# Note: Firebase Functions scheduler is not used with Google Cloud Functions
# Cloud Scheduler is set up separately in the deployment script

# Import our services
from services.firestore_service import FirestoreService
from services.email_service import EmailService
from services.cloud_logging_service import CloudLoggingService
from services.extension_store_service import ExtensionStoreService
from services.report_service import get_records_for_period, generate_report_data

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

analysis_core = AnalysisCore(enable_database=True)


@functions_framework.http
def vegassist_analyze(request: Request):
    """
    Cloud Function entry point for Vegan Confirmed analysis

    Args:
        request: HTTP request object

    Returns:
        HTTP response with analysis results
    """
    # Set CORS headers
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        if request.method == "POST" and request.path == "/api/analyze":
            # Get client IP (Cloud Functions specific)
            client_ip = request.headers.get("X-Forwarded-For", "unknown")
            if client_ip:
                client_ip = client_ip.split(",")[0].strip()

            # Get user agent
            user_agent = request.headers.get("User-Agent")

            # Parse request data
            data = request.get_json()
            if not data:
                return (jsonify({"error": "No data provided"}), 400, headers)

            # Use the analysis core
            response_data, status_code = analysis_core.analyze_page(
                data, client_ip, user_agent
            )
            return (jsonify(response_data), status_code, headers)

        if request.method == "POST" and request.path == "/api/feedback":
            data = request.get_json()
            if not data:
                return (jsonify({"error": "No data provided"}), 400, headers)

            response_data, status_code = analysis_core.record_feedback(data)
            return (jsonify(response_data), status_code, headers)

        # Handle unknown endpoints
        return (jsonify({"error": "Not found"}), 404, headers)

    except Exception as e:
        logger.error(f"Error in Cloud Function: {e}")
        return (jsonify({"error": "Internal server error"}), 500, headers)


@functions_framework.http
def generate_weekly_report(request):
    """
    Cloud Function to generate and send weekly usage reports

    Private; Cloud Scheduler calls it weekly with an OIDC token.

    Expected environment variables:
    - SENDER_EMAIL: Gmail address to send from
    - SENDER_PASSWORD: Gmail app password
    - RECIPIENT_EMAIL: Email address to send reports to
    """

    try:
        # Initialize services
        firestore_service = FirestoreService()
        email_service = EmailService()

        # Initialize Cloud Logging service for page visit stats
        cloud_logging_service = None
        try:
            cloud_logging_service = CloudLoggingService()
            logger.info("Cloud Logging service initialized for page visit stats")
        except Exception as e:
            logger.warning(f"Could not initialize Cloud Logging service: {e}")

        # Calculate date range for the past week
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        logger.info(f"Generating weekly report from {start_date} to {end_date}")

        # Get all records from the past week
        all_records = get_records_for_period(firestore_service, start_date, end_date)

        # Generate report data
        report_data = generate_report_data(
            all_records,
            start_date,
            end_date,
            cloud_logging_service,
            ExtensionStoreService(firestore_service.db),
        )

        # Send email report
        success = email_service.send_weekly_report(report_data)

        # The report goes by email only, never in the response.
        if success:
            logger.info("Weekly report sent successfully")
            return {
                "status": "success",
                "message": "Weekly report sent successfully",
            }, 200
        else:
            logger.error("Failed to send weekly report")
            return {"status": "error", "message": "Failed to send weekly report"}, 500

    except Exception as e:
        logger.error(f"Error generating weekly report: {e}")
        return {"status": "error", "message": str(e)}, 500


# Note: The scheduled_weekly_report function has been removed
# The generate_weekly_report function is now used for both manual and scheduled execution
# Cloud Scheduler triggers the generate_weekly_report function via HTTP POST
