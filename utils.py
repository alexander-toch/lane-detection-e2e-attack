def dummy_env():
    env = MetaDriveEnv({"use_render": False, "image_observation": False})
    try:
        env.reset()
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([0, 1])
    except:
        print("Error happens in Bullet physics world !")
        sys.exit()
    else:
        print("Bullet physics world is launched successfully!")
    finally:
        env.close()
