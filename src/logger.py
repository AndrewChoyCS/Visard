import os
import logging

class Logger():
    def __init__(self):
        self.output_dir = 'logger'
        self.setup_logger()

    def setup_logger(self):
        """Set up logger for pipeline."""
        self.logger = logging.getLogger('PipelineLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers for logging to both console and file
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # Ensure output directory exists
        
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(self.output_dir, 'pipeline_log.txt'))
        
        # Set log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)

    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)

    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)
