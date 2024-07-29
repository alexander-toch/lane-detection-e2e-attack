from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib, FrameBufferProperties

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import Semantics, CameraTagStateKey
from metadrive.engine.core.terrain import Terrain

class LaneCamera(BaseCamera):
    CAM_MASK = CamMask.SemanticCam

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        buffer_props = FrameBufferProperties()
        buffer_props.set_rgba_bits(8, 8, 8, 0)
        buffer_props.set_depth_bits(24)
        buffer_props.set_force_hardware(True)
        buffer_props.set_multisamples(16)
        buffer_props.set_srgb_color(False)
        buffer_props.set_stereo(False)
        buffer_props.set_stencil_bits(0)
        super(LaneCamera, self).__init__(engine, cuda, buffer_props)

    def _setup_effect(self):
        """
        Use tag to apply color to different object class
        Returns: None

        """
        # setup camera
        cam = self.get_cam().node()
        cam.setTagStateKey(CameraTagStateKey.Semantic)

        c = Semantics.LANE_LINE.color

        cam.setTagState(
            Semantics.TERRAIN.label, Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain_semantics_lane.frag.glsl")
        )

        cam.setTagState(Semantics.SKY.label, RenderState.make(
            ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
            ColorAttrib.makeFlat((0.0, 0.0, 0.0, 1)), 1
        ))

        cam.setTagState(Semantics.SIDEWALK.label, RenderState.make(
            ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
            ColorAttrib.makeFlat((0.0, 0.0, 0.0, 1)), 1
        ))

        cam.setTagState(
            Semantics.LANE_LINE.label,
            RenderState.make(
                ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
                ColorAttrib.makeFlat((c[0] / 255, c[1] / 255, c[2] / 255, 1)), 1
            )
        )
