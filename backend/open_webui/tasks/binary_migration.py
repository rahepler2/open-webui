"""
Background task scheduler for automated binary quantization migration
"""
import asyncio
import logging
from typing import Optional
import threading
import time
from datetime import datetime, timedelta

from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.retrieval.vector.dbs.pgvector import PgvectorClient
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))


class BinaryMigrationScheduler:
    """Scheduler for automated background binary quantization migrations"""
    
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.check_interval = 3600  # Check every hour
        self.last_check = None

    def start(self):
        """Start the background migration scheduler"""
        if self.running:
            log.debug("Migration scheduler already running")
            return
            
        # Only start if using pgvector
        if not isinstance(VECTOR_DB_CLIENT, PgvectorClient):
            log.debug("Migration scheduler only supports pgvector, skipping")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        log.info("Binary migration scheduler started")

    def stop(self):
        """Stop the background migration scheduler"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        log.info("Binary migration scheduler stopped")

    def _run_scheduler(self):
        """Main scheduler loop running in background thread"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Only check if enough time has passed
                if (self.last_check is None or 
                    current_time - self.last_check >= timedelta(seconds=self.check_interval)):
                    
                    self.last_check = current_time
                    self._perform_migration_check()
                    
                # Sleep for a shorter interval to allow responsive shutdown
                time.sleep(60)  # Check every minute if we should run migration
                
            except Exception as e:
                log.exception(f"Error in migration scheduler loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _perform_migration_check(self):
        """Perform the actual migration check"""
        try:
            if isinstance(VECTOR_DB_CLIENT, PgvectorClient):
                log.debug("Running automated migration check")
                VECTOR_DB_CLIENT.run_automated_migration_check()
            else:
                log.debug("Skipping migration check - not using pgvector")
                
        except Exception as e:
            log.exception(f"Migration check failed: {e}")

    def force_migration_check(self):
        """Force an immediate migration check (for testing/admin use)"""
        try:
            log.info("Forcing immediate migration check")
            self._perform_migration_check()
        except Exception as e:
            log.exception(f"Forced migration check failed: {e}")


# Global scheduler instance
_migration_scheduler: Optional[BinaryMigrationScheduler] = None


def get_migration_scheduler() -> BinaryMigrationScheduler:
    """Get the global migration scheduler instance"""
    global _migration_scheduler
    if _migration_scheduler is None:
        _migration_scheduler = BinaryMigrationScheduler()
    return _migration_scheduler


def start_migration_scheduler():
    """Start the global migration scheduler"""
    scheduler = get_migration_scheduler()
    scheduler.start()


def stop_migration_scheduler():
    """Stop the global migration scheduler"""
    global _migration_scheduler
    if _migration_scheduler:
        _migration_scheduler.stop()
        _migration_scheduler = None


def force_migration_check():
    """Force an immediate migration check"""
    scheduler = get_migration_scheduler()
    scheduler.force_migration_check()