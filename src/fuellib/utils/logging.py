"""Logger utility module for setting up and configuring logging."""

import logging


class CustomFormatter(logging.Formatter):
    # ANSI escape sequences for text formatting
    grey = "\x1b[38;20m"
    red = "\x1b[31;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"

    date_fmt = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        log_color = {
            logging.DEBUG: self.blue,
            logging.INFO: self.grey,
            logging.WARNING: self.yellow,
            logging.ERROR: self.red,
            logging.CRITICAL: self.bold + self.red,
        }.get(record.levelno, self.reset)

        # Apply color formatting dynamically to the levelname or the whole message
        formatter = logging.Formatter(
            f"[%(asctime)s] {log_color}%(levelname)s:{self.reset} %(message)s\n",
            datefmt=self.date_fmt,
        )
        return formatter.format(record)


# Set up logging
logger = logging.getLogger(__name__)
# Console output handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter())

logging.basicConfig(level=logging.INFO, handlers=[console_handler])
