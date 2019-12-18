import logging
import time


def logging_config(logger: logging.Logger, level: int = 20) -> logging.Logger:
    '''
    configure logging
    '''
    # the line below does not work. the logging objects are singletons.
    # local_logger = copy.deepcopy(logger)

    logger.setLevel(level)

    # console handler
    console_handler = logging.StreamHandler()

    # this handler shall respond to all logging levels.
    # set the logging level in
    console_handler.setLevel(logging.DEBUG)

    # formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # set time to UTC
    formatter.converter = time.gmtime
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)

    logger.debug('configured logger, time is logged in UTC')
    return logger
