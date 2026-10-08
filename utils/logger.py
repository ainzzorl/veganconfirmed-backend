import logging
import os
from datetime import datetime


def setup_logger():
    """Setup logging configuration for the application"""

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler
            logging.FileHandler(
                f'logs/vegassist_{datetime.now().strftime("%Y%m%d")}.log'
            ),
        ],
    )

    # Set specific log levels for external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")
