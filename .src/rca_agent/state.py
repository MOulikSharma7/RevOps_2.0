from typing import TypedDict

class RCAState(TypedDict):
    exception_error:str
    log_error : str
    error_type : str
    root_cause : str
    log_file_path : str  # added (get this value from orchestrator)
