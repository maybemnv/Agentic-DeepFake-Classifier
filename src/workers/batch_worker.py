"""
Batch Worker
Background worker for processing batch jobs.
"""

import asyncio
import logging
from datetime import datetime
from src.core import get_logger, settings
from src.api.schemas import JobStatusEnum

logger = get_logger(__name__)


# In-memory job queue (use Redis/Celery in production)
job_queue: asyncio.Queue = asyncio.Queue()
batch_jobs: dict[str, dict] = {}


async def process_job(job_id: str, job_data: dict):
    """Process a single batch job."""
    logger.info(f"Processing batch job: {job_id}")

    batch_jobs[job_id]["status"] = JobStatusEnum.PROCESSING
    batch_jobs[job_id]["created_at"] = datetime.utcnow()

    try:
        # Process videos here
        # This is a placeholder - actual implementation would:
        # 1. Download/fetch videos
        # 2. Run analysis
        # 3. Store results

        batch_jobs[job_id]["status"] = JobStatusEnum.COMPLETED
        batch_jobs[job_id]["completed_at"] = datetime.utcnow()

        logger.info(f"Batch job completed: {job_id}")

    except Exception as e:
        logger.exception(f"Batch job failed: {job_id}")
        batch_jobs[job_id]["status"] = JobStatusEnum.FAILED
        batch_jobs[job_id]["errors"].append(str(e))


async def worker_loop():
    """Main worker loop."""
    logger.info("Worker started")

    while True:
        try:
            job_id, job_data = await job_queue.get()
            await process_job(job_id, job_data)
            job_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Worker error: {e}")


async def submit_job(job_id: str, job_data: dict):
    """Submit a job to the queue."""
    await job_queue.put((job_id, job_data))
    logger.info(f"Job submitted: {job_id}")


def main():
    """Run the worker."""
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
