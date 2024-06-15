# Main file for running the simulator with one or more scanarios

from jsonargparse import CLI
from metadrive_bridge import MetaDriveBridge, Settings

if __name__ == "__main__":
    setings = CLI(Settings, as_positional=False)
    bridge = MetaDriveBridge(setings)
    bridge.run()
