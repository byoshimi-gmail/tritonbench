import pynvml

class PowerManager():
    def __init__(self) -> None:
        pass

    def start(self) -> None:
        pynvml.nvmlInit()
    
    def stop(self) -> None:
        pynvml.nvmlShutdown()
