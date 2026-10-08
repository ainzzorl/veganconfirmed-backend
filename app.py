from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
from analysis_core import AnalysisCore
from db_viewer import register_db_viewer
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": "*",
        }
    },
)  # Enable CORS for all routes with full permissions

# Initialize analysis core
analysis_core = AnalysisCore()

# Browsable view of the stored API calls. Only registered here, on the local
# entry point: it is unauthenticated, so it also refuses any request that did
# not come from this machine directly. See db_viewer.py.
register_db_viewer(app, analysis_core.database_service)


def _client_ip() -> str:
    """Resolve the client IP, honouring the usual proxy headers."""
    if request.headers.get("X-Forwarded-For"):
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    elif request.headers.get("X-Real-IP"):
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
    return request.remote_addr or "unknown"


@app.route("/api/analyze", methods=["POST"])
def analyze_page():
    """Analyze any page: a shopping item, a restaurant menu, or neither.

    The single analysis endpoint. Extension versions predating the unified
    flow post here too, and still read the product fields they expect off the
    top level of the response.
    """
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Use the analysis core
        response_data, status_code = analysis_core.analyze_page(
            data, _client_ip(), request.headers.get("User-Agent")
        )
        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"Error analyzing page: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/feedback", methods=["POST"])
def record_feedback():
    """Record what a user made of an analysis, by its ``analysis_id``.

    Takes no client IP or user agent, unlike analyze: the record being rated
    already holds them for the request that produced it.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        response_data, status_code = analysis_core.record_feedback(data)
        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5555))
    debug = os.environ.get("FLASK_ENV") == "development"

    logger.info(f"Starting Vegan Confirmed backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
