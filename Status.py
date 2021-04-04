from enum import Enum

class Status(Enum):
    READY = "ready"
    NEW = "new"
    TRAINING = "training"
    FAULT = "fault"
