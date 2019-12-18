import logging

from melanoma_classification.utils import (
    logging_configuration
)


def test_logging(caplog):
    logger = logging.getLogger()
    logger = logging_configuration.logging_config(logger)
    test_message = 'hello, this is a test'
    logger.info(test_message)
    print(caplog)
    messages = [log.message for log in caplog.records]
    assert(test_message in messages)
    # assert('hallo' in messages) # This shall give an error.
