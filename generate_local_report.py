#!/usr/bin/env python3
"""
Generate and view weekly report locally

Usage:
    python generate_local_report.py              # Generate report for the past 7 days
    python generate_local_report.py --days 14   # Generate report for the past 14 days
    python generate_local_report.py --no-open   # Generate without opening in browser
    python generate_local_report.py -o report.html  # Specify output file
"""

import argparse
import os
import sys
import webbrowser
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from services.firestore_service import FirestoreService
from services.cloud_logging_service import CloudLoggingService
from services.extension_store_service import ExtensionStoreService
from services.report_service import (
    get_records_for_period,
    generate_report_data,
    save_report_locally,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and view weekly usage report locally"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to include in the report (default: 7)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated based on date range)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open the report in browser after generating",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🌱 Vegan Confirmed - Local Report Generator")
    print("=" * 60)

    try:
        # Initialize services
        print("\nInitializing services...")
        firestore_service = FirestoreService()

        # Initialize Cloud Logging service for page visit stats
        cloud_logging_service = None
        try:
            cloud_logging_service = CloudLoggingService()
            print("Cloud Logging service initialized for page visit stats")
        except Exception as e:
            print(f"⚠️  Could not initialize Cloud Logging service: {e}")
            print("Page visit stats will not be included in the report.")

        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)

        # Get records
        records = get_records_for_period(firestore_service, start_date, end_date)

        if not records:
            print("\n⚠️  No records found for the specified period.")
            print("The report will be generated but will show zero data.")

        # Generate report data
        report_data = generate_report_data(
            records,
            start_date,
            end_date,
            cloud_logging_service,
            ExtensionStoreService(firestore_service.db),
        )

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Report Summary")
        print("=" * 60)
        print(f"  Period: {report_data['week_start']} to {report_data['week_end']}")
        print(f"  Total Calls: {report_data['total_calls']}")
        print(f"  Unique Clients: {report_data['unique_clients']}")
        print(
            f"  Success Rate: {report_data['success_rate']:.1f}% "
            f"({report_data['successful_calls']} answered, "
            f"{report_data['failed_calls']} failed)"
        )
        print(f"  Item Analyses: {report_data['item_stats']['calls']}")
        print(f"    Vegan: {report_data['vegan_breakdown']['vegan']}")
        print(f"    Non-Vegan: {report_data['vegan_breakdown']['non_vegan']}")
        print(
            f"  Menu Analyses: {report_data['menu_stats']['calls']} "
            f"({report_data['menu_stats']['dishes']} dishes)"
        )
        for service, stats in report_data["provider_stats"].items():
            print(
                f"  Provider {service}: {stats['calls']} calls, "
                f"{stats['read_tokens']:,} read / {stats['write_tokens']:,} write tokens"
            )
        call_stats = report_data["call_stats"]
        print(
            f"  Tokens: {call_stats['read_tokens']:,} read, "
            f"{call_stats['write_tokens']:,} write, "
            f"{call_stats['total_tokens']:,} total"
        )

        # Print page visit stats if available
        page_stats = report_data.get("page_visit_stats")
        if page_stats and not page_stats.get("error"):
            print("-" * 60)
            print("📄 Website Page Visits (Firebase Hosting)")
            print(f"  Total Page Views: {page_stats.get('total_page_views', 0)}")
            print(f"  Unique Visitors: {page_stats.get('unique_visitors', 0)}")

        extension_users = report_data.get("extension_users")
        if extension_users and not extension_users.get("error"):
            print("-" * 60)
            print("🧩 Extension Users")
            for store, stats in extension_users["stores"].items():
                print(f"  {store}: {stats['users']} (previous: {stats['previous']})")
        print("=" * 60)

        # Save report locally
        print("\nGenerating HTML report...")
        output_path = save_report_locally(report_data, args.output)

        if output_path:
            # Convert to absolute path for display and browser
            abs_path = os.path.abspath(output_path)
            print(f"\n✅ Report saved to: {abs_path}")

            # Open in browser unless --no-open specified
            if not args.no_open:
                print("Opening report in browser...")
                webbrowser.open(f"file://{abs_path}")
        else:
            print("\n❌ Failed to save report")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "credentials" in str(e).lower() or "authentication" in str(e).lower():
            print("\n💡 Hint: Make sure you're authenticated with Google Cloud:")
            print("   Run: gcloud auth application-default login")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
