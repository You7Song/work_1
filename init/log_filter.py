import logging

class IgnoreSpecificMessageFilter(logging.Filter):
    """
    自定义日志过滤器，用于过滤掉包含特定消息的日志。
    """
    def __init__(self, message):
        super().__init__()
        self.message = message

    def filter(self, record):
        """
        检查日志消息是否包含特定内容。
        如果包含，则返回 False
        """
        return self.message not in record.getMessage()