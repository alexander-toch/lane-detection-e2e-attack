from metadrive.component.sensors.rgb_camera import RGBCamera
from typing import Union
import numpy as np
from panda3d.core import NodePath
from metadrive.constants import CamMask
_cuda_enable = True
try:
    import cupy as cp
    from OpenGL.GL import GL_TEXTURE_2D  # noqa F403
    from cuda import cudart
    from cuda.cudart import cudaGraphicsRegisterFlags
    from panda3d.core import GraphicsOutput, Texture, GraphicsStateGuardianBase, DisplayRegionDrawCallbackData
except ImportError:
    _cuda_enable = False
from panda3d.core import Vec3

from metadrive import constants

POSITION = (0., 0.8, 15.)
HPR = (90., 0., 0.)

class TopDownCamera(RGBCamera):
    def perceive(
        self, to_float=True, new_parent_node: Union[NodePath, None] = None, position=None, hpr=None
    ) -> np.ndarray:
        if new_parent_node:
            if position is None:
                position = POSITION
            if hpr is None:
                hpr = HPR

            # return camera to original state
            original_object = self.cam.getParent()
            original_hpr = self.cam.getHpr()
            original_position = self.cam.getPos()

            # reparent to new parent node
            self.cam.reparentTo(new_parent_node)

        if position is None:
            position = POSITION
        if hpr is None:
            hpr = HPR

         # relative position
        assert len(position) == 3, "The first parameter of camera.perceive() should be a BaseObject instance " \
                                    "or a 3-dim vector representing the (x,y,z) position."
        self.cam.setPos(Vec3(*position))
        assert len(hpr) == 3, "The hpr parameter of camera.perceive() should be  a 3-dim vector representing " \
                                "the heading/pitch/roll."
        self.cam.setHpr(Vec3(*hpr))

        if new_parent_node:
            self.engine.taskMgr.step()

        if self.enable_cuda:
            assert self.cuda_rendered_result is not None
            ret = self.cuda_rendered_result[..., :self.num_channels][..., ::-1][::-1]
        else:
            ret = self.get_rgb_array_cpu()

        if new_parent_node:
            # return camera to original objects
            self.cam.reparentTo(original_object)
            self.cam.setHpr(original_hpr)
            self.cam.setPos(original_position)

        if not to_float:
            return ret.astype(np.uint8, copy=False, order="C")
        else:
            return ret / 255