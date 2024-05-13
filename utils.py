import sys
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

def dummy_env():
    env = MetaDriveEnv({"use_render": False, "image_observation": True, "sensors": {"rgb_camera": (RGBCamera, 100, 100)}})
    try:
        env.reset()
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    except:
        print("Error happens in Bullet physics world !")
        sys.exit()
    else:
        print("Bullet physics world is launched successfully!")
    finally:
        env.close()
