from typing import TypedDict, Optional, List

class NbaState(TypedDict):
    inputs : List[str]
    root_cause : str
    suggested_fix : str

