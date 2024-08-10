# Main file for running the simulator with one or more scanarios

from metadrive_bridge import MetaDriveBridge, Settings

if __name__ == "__main__":
    setings = Settings()
    bridge = MetaDriveBridge(setings)
    bridge.process_training_data()
