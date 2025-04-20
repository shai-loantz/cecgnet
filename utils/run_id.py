from utils.logger import logger


def get_run_id() -> str:
    try:
        with open('RUN_ID', 'r') as fh:
            return fh.read()
    except Exception:
        logger.exception('Could not get run id. Using placeholder')
        return 'RUN_ID_PLACEHOLDER'


def set_run_id(run_id: str) -> None:
    logger.info(f'Setting {run_id=}')
    with open('RUN_ID', 'w') as fh:
        fh.write(run_id)
