
import logging, os
from datetime import datetime

class HarinLogger:
    def __init__(self, log_dir='logs', level=logging.INFO):
        os.makedirs(log_dir, exist_ok=True)
        fname = datetime.utcnow().strftime('%Y%m%d') + '.log'
        path = os.path.join(log_dir, fname)
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(path, encoding='utf-8'),
                      logging.StreamHandler()]
        )
        self.logger = logging.getLogger('harin')
    def info(self,msg): self.logger.info(msg)
    def debug(self,msg): self.logger.debug(msg)
    def warning(self,msg): self.logger.warning(msg)
    def error(self,msg): self.logger.error(msg)
