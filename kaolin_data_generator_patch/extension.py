import os
import re
import json
import random
import asyncio
import posixpath
import threading
import webbrowser
from queue import Queue
import glob
from functools import partial
import pathlib


import carb
import omni.ext
import omni.syntheticdata as sd
from omni import ui
from carb import settings
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Vt, Gf, Sdf, Tf, Semantics
import numpy as np
import omni.syntheticdata as sd
from omni.kit.pointcloud_generator import PointCloudGenerator

from kaolin_app.research import utils
from .utils import (
    delete_sublayer,
    omni_shader,
    bottom_to_elevation,
    save_to_log,
    save_numpy_array,
    save_image,
    save_pointcloud,
    wait_for_loaded,
)
from .sensors import _build_ui_sensor_selection
from .ui import build_component_frame
from .dr_components import sample_component


_extension_instance = None


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE = os.path.join(FILE_DIR, ".cache")
EXTENSION_NAME = "Data Generator"
SCENE_PATH = "/World/visualize"
NUM_WORKERS = 10
VALID_EXTENSIONS = ["*.usd", "*.usda", "*.usdc"]
RENDERERS = ["RaytracedLighting", "PathTracing"]
CAMERAS = ["UniformSampling", "Trajectory"]
TRAJ_OPTIONS = ["Spiral", "CustomJson"]
DEMO_URL = "https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin/user_manual.html#data-generator"
MAX_RESOLUTION = {"width": 7680, "height": 4320}
MIN_RESOLUTION = {"width": 1024, "height": 1024}
DR_COMPONENTS = [
    "LightComponent",
    "MovementComponent",
    "RotationComponent",
    "ColorComponent",
    "TextureComponent",
    "MaterialComponent",
    "VisibilityComponent",
]


class KaolinDataGeneratorError(Exception):
    pass


class IOWorkerPool:
    def __init__(self, num_workers: int):
        self.save_queue = Queue()

        for _ in range(num_workers):
            t = threading.Thread(target=self._do_work)
            t.start()

    def add_to_queue(self, data: object):
        self.save_queue.put(data)

    def _do_work(self):
        while True:
            fn = self.save_queue.get(block=True)
            fn()
            self.save_queue.task_done()


class Extension(omni.ext.IExt):
    def __init__(self):
        self.root_dir = None
        self._ref_idx = 0
        self._filepicker = None
        self._outpicker = None
        self._configpicker = None
        self._jsonpicker = None
        self.camera = None
        self._preset_layer = None
        self.dr_components = {}
        self.asset_list = None
        self._progress_tup = None
        self.option_frame = None
        self.config = {}
        self.start_config = {}

    def get_name(self):
        return EXTENSION_NAME

    def on_startup(self, ext_id: str):
        global _extension_instance
        _extension_instance = self

        self._settings = carb.settings.get_settings()
        self.progress = None
        self._context = omni.usd.get_context()
        self._window = ui.Window(EXTENSION_NAME, width=500, height=500)
        self._menu_entry = omni.kit.ui.get_editor_menu().add_item(
            f"Window/Kaolin/{EXTENSION_NAME}", self._toggle_menu, toggle=True, value=True
        )
        self._preview_window = ui.Window("Preview", width=500, height=500)
        self._preview_window.deferred_dock_in("Property")
        self._preview_window.visible = False
        self._filepicker = omni.kit.window.filepicker.FilePickerDialog(
            "Select Asset(s)",
            click_apply_handler=lambda f, d: self._on_filepick(f, d),
            apply_button_label="Open",
            item_filter_options=["usd", "usda", "usdc"],
        )
        self._filepicker.hide()
        self._outpicker = omni.kit.window.filepicker.FilePickerDialog(
            "Select Output Directory",
            click_apply_handler=lambda _, x: self._on_outpick(x),
            apply_button_label="Select",
            enable_filename_input=False,
        )
        self._outpicker.hide()
        self._configpicker = omni.kit.window.filepicker.FilePickerDialog(
            "Import Preset",
            click_apply_handler=self._on_load_config,
            apply_button_label="Open",
            item_filter_options=["usda"],
        )
        self._configpicker.hide()
        self._jsonpicker = omni.kit.window.filepicker.FilePickerDialog(
            "Import Json trajectory file",
            click_apply_handler=lambda f, d: asyncio.ensure_future(
                self._import_trajectory_from_json(posixpath.join(d, f))
            ),
            apply_button_label="Open",
            item_filter_fn=self._on_filter_json,
        )
        self._jsonpicker.hide()
        self._configsaver = omni.kit.window.filepicker.FilePickerDialog(
            "Save Preset As...",
            click_apply_handler=self._on_save_config,
            apply_button_label="Save",
            item_filter_options=["usda"],
        )
        cache = {}
        if not os.path.exists(CACHE):
            os.makedirs(CACHE, exist_ok=True)
        if posixpath.exists(os.path.join(CACHE, ".log")):
            with open(os.path.join(CACHE, ".log"), "r") as f:
                cache = json.load(f)
        self._cache = cache
        self._hide_filepickers()
        self.start_config = self._set_start_config()
        self.presets = [str(pathlib.Path(p).as_posix()) for p in glob.glob(posixpath.join(FILE_DIR, "presets/*.usda"))]
        self.stage_events_sub = self._context.get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)
        self.sdv = sd.Extension.get_instance()
        self._vp_iface = omni.kit.viewport.get_viewport_interface()
        self.timeline = omni.timeline.get_timeline_interface()
        self._build_ui()

    def on_shutdown(self):
        global _extension_instance
        _extension_instance = None

        if self._preset_layer:
            delete_sublayer(self._preset_layer)
        self.progress = None
        if self._window:
            del self._window
        if self._filepicker:
            self._filepicker = None
        if self._outpicker:
            self._outpicker = None
        if self._configpicker:
            self._configpicker = None
        if self._jsonpicker:
            self._jsonpicker = None

    def _toggle_menu(self, *args):
        self._window.visible = not self._window.visible

    def clear(self):
        if self._preset_layer:
            delete_sublayer(self._preset_layer)

        # reset resolution
        self._settings.set("/app/renderer/resolution/width", self.start_config["width"])
        self._settings.set("/app/renderer/resolution/height", self.start_config["height"])

        # reset rendering mode
        self._settings.set("/rtx/rendermode", self.start_config["renderer"])
        self._settings.set("/rtx-defaults/pathtracing/clampSpp", self.start_config["clampSpp"])
        self._settings.set("/rtx-defaults/pathtracing/totalSpp", self.start_config["totalSpp"])
        self._settings.set("/rtx/post/aa/op", self.start_config["aa"])

    def _on_stage_event(self, e):
        pass

    def _reset(self):
        self._ref_idx = 0
        self.asset_list = None

    def _show_filepicker(self, filepicker, default_dir: str = "", default_file: str = ""):
        cur_dir = filepicker.get_current_directory()
        show_dir = cur_dir if cur_dir else default_dir
        filepicker.show(show_dir)
        filepicker.set_filename(default_file)

    def _hide_filepickers(self):
        # Hide all filepickers
        self._configsaver.hide()
        self._filepicker._click_cancel_handler = self._filepicker.hide()
        self._outpicker._click_cancel_handler = self._outpicker.hide()
        self._jsonpicker._click_cancel_handler = self._jsonpicker.hide()
        self._configpicker._click_cancel_handler = self._configpicker.hide()
        self._configsaver._click_cancel_handler = self._configsaver.hide()

    def _set_start_config(self):
        return {
            "width": self._settings.get("/app/renderer/resolution/width"),
            "height": self._settings.get("/app/renderer/resolution/height"),
            "renderer": self._settings.get("/rtx/rendermode"),
            "clampSpp": self._settings.get("/rtx-defaults/pathtracing/clampSpp"),
            "totalSpp": self._settings.get("/rtx/pathtracing/totalSpp"),
            "aa": self._settings.get("/rtx/post/aa/op"),
        }

    def _on_filter_json(self, item: omni.kit.widget.filebrowser.filesystem_model.FileSystemItem):
        file_exts = ["json", "JSON"]
        for fex in file_exts:
            if item.name.endswith(fex) or item.is_folder:
                return True

    async def _import_trajectory_from_json(self, path: str):
        """ Import a trajectory from a JSON file in a predefined format. """
        trajectory = self._on_load_json(path)
        self.config["jsonpath"] = path
        assert isinstance(trajectory, list)
        assert len(trajectory) > 0

        # add trajectory prim
        stage = omni.usd.get_context().get_stage()
        timestamp_prim = stage.DefinePrim(f"{SCENE_PATH}/timestamp", "Xform")
        trajectory_rig = stage.DefinePrim(f"{timestamp_prim.GetPath()}/rig", "Xform")
        UsdGeom.Xformable(trajectory_rig).ClearXformOpOrder()
        UsdGeom.Xformable(trajectory_rig).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        UsdGeom.Xformable(trajectory_rig).AddOrientOp()

        # Set translation and orientation according to trajectory
        origins, scales, orientations = [], [], []
        for idx, entry in enumerate(trajectory):
            # Set camera based on time, translation, quaternion in the json file.
            trans, quaternion, time = entry["t"], entry["q"], entry["time"]
            # The JSON format has different camera coordinate system conventions:
            # +X points right, +Y points down, camera faces in +Z.
            # Compared to Kit's conventions:
            # +X points right, -Y points down, camera faces in -Z.
            # So the Y and Z axes need to be flipped, and orientations need to be
            # rotated around X by 180 degrees for the coordinate systems to match.
            trans[1] = -trans[1]  # Flip Y
            trans[2] = -trans[2]  # Flip Z
            # Set translation and orientations according to time.
            trajectory_rig.GetAttribute("xformOp:translate").Set(Gf.Vec3d(trans), time=time)

            # Both the JSON format and Gf.Quatd use a "scalar first" ordering.
            # Flip Y and Z axes.
            quaternion[2] = -quaternion[2]
            quaternion[3] = -quaternion[3]
            trajectory_rig.GetAttribute("xformOp:orient").Set(Gf.Quatf(*quaternion), time=time)

            # Use prev and curr translation to generate a trajectory vis as PointInstancer
            orientation = Gf.Quath(*quaternion).GetNormalized()
            orientations.append(orientation)
            origins.append(Gf.Vec3d(trans))
            scales.append([1.0, 1.0, 1.0])

        # Define prim for visualization, each component will be a cone (like 3d vector)
        cone_height = 0.03
        proto_prim = stage.DefinePrim(f"{SCENE_PATH}/proto", "Xform")
        proto_prim.GetAttribute("visibility").Set("invisible")
        cone_rig = stage.DefinePrim(f"{proto_prim.GetPath()}/cone", "Xform")
        cone = UsdGeom.Cone.Define(stage, (f"{cone_rig.GetPath()}/cone"))
        cone.GetRadiusAttr().Set(0.01)
        cone.GetHeightAttr().Set(cone_height)
        cone.GetAxisAttr().Set("Z")

        # cone rig
        UsdGeom.Xformable(cone_rig).ClearXformOpOrder()
        UsdGeom.Xformable(cone_rig).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set((0.0, cone_height / 2, 0.0))
        # Setup point instancer
        instancer_prim = stage.DefinePrim(f"{SCENE_PATH}/Viz", "PointInstancer")
        instancer = UsdGeom.PointInstancer(instancer_prim)
        assert instancer
        instancer.CreatePrototypesRel().SetTargets([cone_rig.GetPath()])
        # Populate point instancer with the calculated scales, positions, and orientations
        instancer.GetPositionsAttr().Set(origins)
        instancer.GetScalesAttr().Set(scales)
        indices = [0] * len(origins)
        instancer.GetProtoIndicesAttr().Set(indices)
        instancer.GetOrientationsAttr().Set(orientations)

        await self._preview_trajectory()

    def _move_camera(self, centre: Gf.Vec3d, azimuth: float, elevation: float, distance: float):
        stage = omni.usd.get_context().get_stage()
        rig = stage.GetPrimAtPath(f"{SCENE_PATH}/CameraRig")
        boom = stage.GetPrimAtPath(f"{rig.GetPath()}/Boom")
        camera = stage.GetPrimAtPath(f"{boom.GetPath()}/Camera")

        UsdGeom.Xformable(rig).ClearXformOpOrder()
        centre_op = UsdGeom.Xformable(rig).AddTranslateOp()
        centre_op.Set(tuple(centre))

        rig_rotate_op = UsdGeom.Xformable(rig).AddRotateXYZOp()
        rig_rotate_op.Set((0.0, azimuth, 0.0))

        UsdGeom.Xformable(boom).ClearXformOpOrder()
        boom_rotate_op = UsdGeom.Xformable(boom).AddRotateXYZOp()
        boom_rotate_op.Set((-elevation, 0.0, 0.0))

        # Reset camera
        UsdGeom.Xformable(camera).ClearXformOpOrder()

        distance_op = UsdGeom.Xformable(camera).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        distance_op.Set((0.0, 0.0, distance))

        UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)

    def _get_value(self, option, default=None):
        if option not in self.config:
            self.config[option] = default
        if self.config[option]["mode"] == 0:
            return self.config[option]["fixed"]
        else:
            v_min, v_max = self.config[option]["random"]
            if isinstance(v_min, list):
                return [random.random() * (v_max_el - v_min_el) + v_min_el for v_min_el, v_max_el in zip(v_min, v_max)]
            else:
                return random.random() * (v_max - v_min) + v_min

    def _set_trajectory_camera_pose(self, cur_frame: int, num_frames: int):
        """
        Calculate the camera pose based on a trajectory, number of frames to generate and current frame
        """
        stage = omni.usd.get_context().get_stage()
        viz_prim = stage.GetPrimAtPath(f"{SCENE_PATH}/Viz")

        # Match transform of visualization prim
        tf = UsdGeom.Xformable(viz_prim).ComputeLocalToWorldTransform(0.0)  # .GetInverse()
        camera_rig = stage.GetPrimAtPath(f"{SCENE_PATH}/CameraRig")
        UsdGeom.Xformable(camera_rig).ClearXformOpOrder()
        UsdGeom.Xformable(camera_rig).AddTransformOp().Set(tf)

        trajectory_rig = stage.GetPrimAtPath(f"{SCENE_PATH}/timestamp/rig")
        translations = trajectory_rig.GetAttribute("xformOp:translate")
        time_samples = translations.GetTimeSamples()

        if num_frames <= 1:
            cur_time = (time_samples[-1] - time_samples[0]) / 2.0
        else:
            cur_time = (time_samples[-1] - time_samples[0]) / (num_frames - 1) * cur_frame
        translate = trajectory_rig.GetAttribute("xformOp:translate").Get(time=cur_time)
        orientation = trajectory_rig.GetAttribute("xformOp:orient").Get(time=cur_time)

        UsdGeom.Xformable(self.camera).ClearXformOpOrder()
        UsdGeom.Xformable(self.camera).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translate)
        UsdGeom.Xformable(self.camera).AddOrientOp().Set(orientation)

    def _get_spiral_camera_pose(self, frame, total_frames):
        """
        Calculate the rotation with respect to X & Y based on the current iteration
        of all the sampling
        """
        distance = self._get_value("distance")
        min_ele, max_ele = tuple(self.config["elevation"]["random"])
        numrot = self.config["num_rotations"]

        if total_frames > 1:
            az_step = 360 * numrot / (total_frames - 1)
            ele_step = (max_ele - min_ele) / (total_frames - 1)
        else:
            az_step = 0
            ele_step = 0
        az = frame * az_step
        ele = min_ele + frame * ele_step
        return az, ele, distance

    def _normalize(self, prim: Usd.Prim):
        prim_range = UsdGeom.Imageable(prim).ComputeLocalBound(0, "default").GetRange()
        range_min = prim_range.GetMin()
        range_max = prim_range.GetMax()
        size = prim_range.GetSize()
        sf = 1.0 / max(size)
        offset = (range_max + range_min) / 2 * sf
        UsdGeom.Xformable(prim).AddTranslateOp().Set(-offset)
        UsdGeom.Xformable(prim).AddScaleOp().Set((sf, sf, sf))

    def _change_up_axis(self, model):
        # TODO type
        self.config["up_axis"] = model.as_int

    def add_semantics(self, prim: Usd.Prim, semantic_label: str):
        if not prim.HasAPI(Semantics.SemanticsAPI):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set("class")
            sem.GetSemanticDataAttr().Set(semantic_label)

    def create_asset_prim(self):
        stage = omni.usd.get_context().get_stage()
        asset_prim = stage.GetPrimAtPath(f"{SCENE_PATH}/Asset")
        if not asset_prim:
            asset_prim = stage.DefinePrim(f"{SCENE_PATH}/Asset", "Xform")
        rig_prim = stage.GetPrimAtPath(f"{asset_prim.GetPath()}/Rig")
        if not rig_prim:
            rig_prim = stage.DefinePrim(f"{asset_prim.GetPath()}/Rig", "Xform")
            UsdGeom.Xformable(rig_prim).AddTranslateOp()
            UsdGeom.Xformable(rig_prim).AddRotateXOp()
        translate_op = rig_prim.GetAttribute("xformOp:translate")
        if not translate_op:
            translate_op = UsdGeom.Xformable(rig_prim).AddTranslateOp()
        translate_op.Set((0.0, 0.0, 0.0))
        rotatex_op = rig_prim.GetAttribute("xformOp:rotateX")
        if not rotatex_op:
            UsdGeom.Xformable(rig_prim).AddRotateXOp()
        ref_prim = stage.DefinePrim(f"{SCENE_PATH}/Asset/Rig/Preview")
        self.add_semantics(ref_prim, "asset")
        return asset_prim

    async def _run(self):
        i = 0
        while i < len(self.asset_list):
            self.progress["bar1"].set_value(i / len(self.asset_list))
            if self.progress["stop_signal"]:
                break
            load_success = False
            # If asset fails to load, remove from list and try the next one
            while not load_success and i < len(self.asset_list):
                carb.log_info(f"[kaolin_app.research.data_generator] Loading asset {self.asset_list[i]}...")
                load_success = await self.load_asset(self.asset_list[i], use_cache=True)
                if not load_success:
                    self.asset_list.pop(i)
                if self.progress["stop_signal"]:
                    break

            for j in range(self.config["renders_per_asset"]):
                self.progress["bar2"].set_value(j / self.config["renders_per_asset"])
                if self.progress["stop_signal"]:
                    break
                app = omni.kit.app.get_app_interface()
                await app.next_update_async()
                await self.render_asset(j, self.config["renders_per_asset"])

                self._preview_window.visible = False
                await self._save_gt(i * self.config["renders_per_asset"] + j)

            i += 1
            self._ref_idx += 1

    async def run(self):
        root_layer = omni.usd.get_context().get_stage().GetRootLayer()
        if len(root_layer.subLayerPaths) == 0 or self._preset_layer != Sdf.Find(root_layer.subLayerPaths[-1]):
            self._on_preset_changed(self.presets[self._preset_model.get_item_value_model().as_int], update_config=False)

        if not self.config["out_dir"]:
            m = self._ui_modal("Output Dir Not Specified", "Please specify an output directory.")
            # TODO Notification
            return

        is_custom_json_mode = (
            self.config["cameramode"] == "Trajectory" and self.config["trajectorymode"] == "CustomJson"
        )
        if is_custom_json_mode and not os.path.exists(self.config.get("jsonpath", "")):
            if not self.config.get("jsonpath"):
                title = "JSON Path Not Specified"
            else:
                title = "Invalid JSON Path Specified"
            m = self._ui_modal(title, "Please specify a valid path to a trajectory JSON file.")
            # TODO Notification
            return

        # Set small camera near plane
        cur_clipping_range = self.camera.GetAttribute("clippingRange").Get()
        self.camera.GetAttribute("clippingRange").Set((0.01, cur_clipping_range[1]))

        # Hide path visualization if exists
        if omni.usd.get_context().get_stage().GetPrimAtPath(f"{SCENE_PATH}/Viz"):
            self._set_visible(f"{SCENE_PATH}/Viz", False)

        # Set SPP per config
        self._settings.set("/rtx/pathtracing/spp", self.config["spp"])

        # Capture scene state
        cur_sel = omni.usd.get_context().get_selection().get_selected_prim_paths()
        display_mode = self._settings.get("/persistent/app/viewport/displayOptions")

        # Clear scene state
        omni.usd.get_context().get_selection().clear_selected_prim_paths()
        self._settings.set("/persistent/app/viewport/displayOptions", 0)

        if self.asset_list is None:
            self.asset_list = await utils.path.get_usd_files_async(self.root_dir)

        self._ui_toggle_visible([self.option_frame, self.progress["block"]])

        # Reset Camera
        if not self.camera.GetAttribute("xformOp:translate"):
            UsdGeom.Xformable(self.camera).AddTranslateOp()
        self.camera.GetAttribute("xformOp:translate").Set((0, 0, 0))
        if not self.camera.GetAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(self.camera).AddRotateXYZOp()
        self.camera.GetAttribute("xformOp:rotateXYZ").Set((0, 0, 0))

        try:
            await self._run()
        except Exception as e:
            raise e
        finally:
            self.progress["stop_signal"] = False
            self._ui_toggle_visible([self.option_frame, self.progress["block"]])

            # Re-apply scene state
            omni.usd.get_context().get_selection().set_selected_prim_paths(cur_sel, True)
            self._settings.set("/persistent/app/viewport/displayOptions", display_mode)
            self._settings.set("/rtx/pathtracing/spp", 1)
            self.camera.GetAttribute("clippingRange").Set((1.0, cur_clipping_range[1]))

            if omni.usd.get_context().get_stage().GetPrimAtPath(f"{SCENE_PATH}/Viz"):
                self._set_visible(f"{SCENE_PATH}/Viz", True)

    async def preview(self):
        root_layer = omni.usd.get_context().get_stage().GetRootLayer()
        if len(root_layer.subLayerPaths) == 0 or self._preset_layer != Sdf.Find(root_layer.subLayerPaths[-1]):
            self._on_preset_changed(self.presets[self._preset_model.get_item_value_model().as_int], update_config=False)
        if self.asset_list is None:
            self.asset_list = await utils.path.get_usd_files_async(self.root_dir)

        # Hide path visualization if exists
        if omni.usd.get_context().get_stage().GetPrimAtPath(f"{SCENE_PATH}/Viz"):
            self._set_visible(f"{SCENE_PATH}/Viz", False)

        success = False
        # draw assets at random. Remove invalid assets if detected.
        while not success and len(self.asset_list) > 0:
            sel = random.randrange(len(self.asset_list))
            success = await self.load_asset(self.asset_list[sel], use_cache=False)
            if not success:
                self.asset_list.pop(sel)
        await self.render_asset(random.randrange(100), 100)

        # ensure material is loaded
        await wait_for_loaded()

        self.sdv.build_visualization_ui(self._preview_window, "Viewport")
        self._preview_window.visible = True

        # Set camera target to facilitate camera control
        viewport = omni.kit.viewport.get_viewport_interface().get_viewport_window()
        viewport.set_camera_target(str(self.camera.GetPath()), 0.0, 0.0, 0.0, True)

    def _add_ref(self, ref_prim, file):
        # Check if file has a default prim - if not, use the first prim
        ref_prim.GetReferences().ClearReferences()
        file_stage = Usd.Stage.Open(file)
        if file_stage.HasDefaultPrim():
            ref_prim.GetPrim().GetReferences().AddReference(file)
        else:
            top_level_prims = file_stage.GetPseudoRoot().GetChildren()
            if len(top_level_prims) == 0:
                raise KaolinDataGeneratorError(f"Asset at {file} appears to be empty")
            root_prim = top_level_prims[0]
            ref_prim.GetPrim().GetReferences().AddReference(file, str(root_prim.GetPath()))
        return True

    async def load_asset(self, path: str, use_cache: bool = False):
        # TODO docstring
        stage = omni.usd.get_context().get_stage()
        ref_prim = stage.GetPrimAtPath(f"{SCENE_PATH}/Asset/Rig/Preview")
        if not ref_prim:
            self.create_asset_prim()
            ref_prim = stage.GetPrimAtPath(f"{SCENE_PATH}/Asset/Rig/Preview")
        self._set_visible(str(ref_prim.GetPath()), True)

        try:
            self._add_ref(ref_prim, path)
        except Tf.ErrorException:
            carb.log_warn(f"Error opening {path}.")
            return False
        except KaolinDataGeneratorError as e:
            carb.log_warn(e.args[0])
            return False

        # set transforms
        UsdGeom.Xformable(ref_prim).ClearXformOpOrder()
        if self.config.get("up_axis", 0):
            UsdGeom.Xformable(ref_prim).AddRotateXOp().Set(-90.0)  # If Z up, rotate about X axis
        if self.config.get("asset_normalize"):
            self._normalize(ref_prim)
        if self.config["asset_override_bottom_elev"]:
            bottom_to_elevation(ref_prim.GetParent(), 0.0)
        else:
            ref_prim.GetParent().GetAttribute("xformOp:translate").Set((0.0, 0.0, 0.0))

        # ensure material is loaded
        await asyncio.sleep(1)
        await wait_for_loaded()

        asset_size = UsdGeom.Imageable(ref_prim).ComputeLocalBound(0, "default").GetRange().GetSize()
        if all([s < 1e-10 for s in asset_size]):
            # Stage is empty, skip asset
            carb.log_warn(f"Asset at {path} appears to be empty.")
            print(
                asset_size,
                ref_prim,
                ref_prim.GetAttribute("visibility").Get(),
                ref_prim.GetMetadata("references").GetAddedOrExplicitItems()[0].assetPath,
            )
            return False

        return True

    async def render_asset(self, cur_frame: int = 0, num_frames: int = 0) -> None:
        # TODO docstring
        self._settings.set("/app/hydraEngine/waitIdle", True)  # Necessary, waitIdle resets itself to false
        stage = omni.usd.get_context().get_stage()
        if not self.camera:
            rig = stage.DefinePrim(f"{SCENE_PATH}/CameraRig", "Xform")
            boom = stage.DefinePrim(f"{rig.GetPath()}/Boom", "Xform")
            self.camera = stage.DefinePrim(f"{boom.GetPath()}/Camera", "Camera")
            self.camera.GetAttribute("clippingRange").Set((1.0, 1000000))

        self._vp_iface.get_viewport_window().set_active_camera(str(self.camera.GetPath()))

        if self.config.get("cameramode") == "Trajectory":
            if self.config["trajectorymode"] == "Spiral":
                centre = self._get_value("centre")
                azimuth, elevation, distance = self._get_spiral_camera_pose(cur_frame, num_frames)
                self._move_camera(centre, azimuth, elevation, distance)
            elif self.config["trajectorymode"] == "CustomJson":
                self._move_camera((0, 0, 0), 0, 0, 0)
                self._set_trajectory_camera_pose(cur_frame, num_frames)
        else:
            centre = self._get_value("centre")
            azimuth = self._get_value("azimuth")
            elevation = self._get_value("elevation")
            distance = self._get_value("distance")
            self._move_camera(centre, azimuth, elevation, distance)

        # Set focal length
        focal_length_defaults = {"fixed": 24.0, "mode": 0, "random": Gf.Vec2f([1.0, 120.0])}
        focal_length = self._get_value("camera_focal_length", focal_length_defaults)
        self.camera.GetAttribute("focalLength").Set(focal_length)

        self.move_asset()
        self.sample_components()
        app = omni.kit.app.get_app_interface()
        await app.next_update_async()  # This next frame await is needed to avoid camera transform remaining in place

    def _get_camera_properties(self):
        width = self._settings.get("/app/renderer/resolution/width")
        height = self._settings.get("/app/renderer/resolution/height")
        tf_mat = np.array(UsdGeom.Xformable(self.camera).ComputeLocalToWorldTransform(0.0).GetInverse()).tolist()
        tf_mat[-1][2] *= 100
        clippingrange = self.camera.GetAttribute("clippingRange").Get()
        clippingrange[0] = 1
        cam_props = {
            "resolution": {"width": width, "height": height},
            "clipping_range": tuple(clippingrange),#tuple(self.camera.GetAttribute("clippingRange").Get()),
            "horizontal_aperture": self.camera.GetAttribute("horizontalAperture").Get(),
            "focal_length": self.camera.GetAttribute("focalLength").Get(),
            "tf_mat": tf_mat,#np.array(UsdGeom.Xformable(self.camera).ComputeLocalToWorldTransform(0.0).GetInverse()).tolist(),
        }
        return cam_props

    def _get_filepath_from_primpath(self, prim_path):
        """ Called to get file path from a prim object. """
        if not prim_path:
            return ""
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if prim:
            metadata = prim.GetMetadata("references")

            if prim and metadata:
                return metadata.GetAddedOrExplicitItems()[0].assetPath
        return ""

    def _get_frame_metadata(
        self, bbox_2d_tight: np.ndarray = None, bbox_2d_loose: np.ndarray = None, bbox_3d: np.ndarray = None
    ):
        frame = {"camera_properties": self._get_camera_properties()}
        if bbox_2d_tight is not None:
            frame["bbox_2d_tight"] = self._get_bbox_2d_data(bbox_2d_tight)
        if bbox_2d_loose is not None:
            frame["bbox_2d_loose"] = self._get_bbox_2d_data(bbox_2d_loose)
        if bbox_3d is not None:
            frame["bbox_3d"] = self._get_bbox_3d_data(bbox_3d)
        ref_prim_path = f"{SCENE_PATH}/Asset/Rig/Preview"
        stage = omni.usd.get_context().get_stage()
        ref_prim = stage.GetPrimAtPath(ref_prim_path)
        tf = np.array(UsdGeom.Xformable(ref_prim).ComputeLocalToWorldTransform(0.0)).tolist()
        ref = self._get_filepath_from_primpath(ref_prim_path)
        if os.path.isfile(self.root_dir):
            rel_ref = os.path.basename(ref)
        else:
            rel_ref = posixpath.relpath(ref, self.root_dir)
        frame["asset_transforms"] = [(rel_ref, tf)]
        json_buffer = bytes(json.dumps(frame, indent=4), encoding="utf-8")
        return json_buffer

    def _get_bbox_2d_data(self, bboxes):
        # TODO type
        bbox_2d_list = []
        for bb_data in bboxes:
            ref = self._get_filepath_from_primpath(bb_data["name"])
            rel_ref = posixpath.relpath(ref, self.root_dir) if ref else ""
            bb_dict = {
                "file": rel_ref,
                "class": bb_data["semanticLabel"],
                "bbox": {a: bb_data[a].item() for a in ["x_min", "y_min", "x_max", "y_max"]},
            }
            bbox_2d_list.append(bb_dict)
        return bbox_2d_list

    def _get_bbox_3d_data(self, bboxes):
        # TODO type
        bbox_3d_list = []
        for bb_data in bboxes:
            ref = self._get_filepath_from_primpath(bb_data["name"])
            rel_ref = posixpath.relpath(ref, self.root_dir) if ref else ""
            bb_dict = {
                "file": rel_ref,
                "class": bb_data["semanticLabel"],
                "bbox": {a: bb_data[a].item() for a in ["x_min", "y_min", "x_max", "y_max", "z_min", "z_max"]},
            }
            bb_dict["transform"] = bb_data["transform"].tolist()
            bbox_3d_list.append(bb_dict)
        return bbox_3d_list

    def move_asset(self):
        stage = omni.usd.get_context().get_stage()

        if self.config["asset_override_bottom_elev"]:
            ref_prim = stage.GetPrimAtPath(f"{SCENE_PATH}/Asset/Rig/Preview")
            bottom_to_elevation(ref_prim.GetParent(), self.config["asset_bottom_elev"])

    async def _save_gt(self, idx: int):
        vp = self._vp_iface.get_viewport_window()
        self._sensors = self.sdv._sensors["Viewport"]
        await sd.sensors.initialize_async(
            vp, [st for _, s in self._sensors.items() if s["enabled"] for st in s["sensors"]]
        )
        io_tasks = []
        img_funcs = {"rgb": partial(sd.sensors.get_rgb, vp), "normals": partial(sd.visualize.get_normals, vp)}
        np_funcs = {
            "depth": partial(sd.sensors.get_depth_linear, vp),
            "instance": partial(sd.sensors.get_instance_segmentation, vp, parsed=(self._sensors["instance"]["mode"])),
            "semantic": partial(sd.sensors.get_semantic_segmentation, vp),
        }

        for sensor, write_fn in img_funcs.items():
            if self._sensors[sensor]["enabled"]:
                filepath = posixpath.join(self.config["out_dir"], f"{idx}_{sensor}.png")
                data = write_fn()
                io_tasks.append(save_image(filepath, data))
                carb.log_info(f"[kaolin.data_generator] Saving {sensor} to {filepath}")

        for sensor, write_fn in np_funcs.items():
            if self._sensors[sensor]["enabled"]:
                filepath = posixpath.join(self.config["out_dir"], f"{idx}_{sensor}.npy")
                data = write_fn()
                io_tasks.append(save_numpy_array(filepath, data))
                carb.log_info(f"[kaolin.data_generator] Saving {sensor} to {filepath}")

        bbox_2d_tight, bbox_2d_loose, bbox_3d = None, None, None
        if self._sensors["bbox_2d_tight"]["enabled"]:
            bbox_2d_tight = sd.sensors.get_bounding_box_2d_tight(vp)
        if self._sensors["bbox_2d_loose"]["enabled"]:
            bbox_2d_loose = sd.sensors.get_bounding_box_2d_loose(vp)
        if self._sensors["bbox_3d"]["enabled"]:
            bbox_3d = sd.sensors.get_bounding_box_3d(vp, parsed=self._sensors["bbox_3d"]["mode"])
        if self._sensors["pointcloud"]["enabled"]:
            pc_gen = PointCloudGenerator()
            pc_gen.stage = omni.usd.get_context().get_stage()
            pc_gen.ref = pc_gen.stage.GetPrimAtPath(f"{SCENE_PATH}/Asset/Rig")
            pc_gen.height_resolution = self._sensors["pointcloud"]["sampling_resolution"]
            pc_gen.width_resolution = self._sensors["pointcloud"]["sampling_resolution"]
            pointcloud = await pc_gen.generate_pointcloud()
            filepath = posixpath.join(self.config["out_dir"], f"{idx}_pointcloud.usd")
            up_axis = ["Y", "Z"][self.config.get("up_axis", 0)]
            io_tasks.append(save_pointcloud(filepath, pointcloud, up_axis))

        filepath = posixpath.join(self.config["out_dir"], f"{idx}_metadata.json")
        frame = self._get_frame_metadata(bbox_2d_tight, bbox_2d_loose, bbox_3d)  # TODO: fix and remove this
        io_tasks.append(omni.client.write_file_async(filepath, frame))
        await asyncio.gather(*io_tasks)

    def sample_components(self):
        # TODO docstring
        for _, components in self.dr_components.items():
            for component in components:
                sample_component(component)

    def _set_visible(self, path: str, value: bool):
        opts = ["invisible", "inherited"]
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if prim and prim.GetAttribute("visibility"):
            prim.GetAttribute("visibility").Set(opts[value])

    def _on_value_changed(self, option, value, idx: int = None, idx_opt=None):
        # TODO type
        has_mode = isinstance(self.config[option], dict)
        if has_mode:
            mode = ["fixed", "random"][self.config[option]["mode"]]
            if idx is not None and idx_opt is not None:
                self.config[option][mode][idx_opt][idx] = value
            elif idx is not None:
                self.config[option][mode][idx] = value
            else:
                self.config[option][mode] = value
        else:
            if idx is not None and idx_opt is not None:
                self.config[option][idx_opt][idx] = value
            elif idx is not None:
                self.config[option][idx] = value
            else:
                self.config[option] = value

    def _on_mode_changed(self, option, model):
        # TODO type
        idx = model.get_item_value_model().get_value_as_int()
        self.config[option]["mode"] = idx
        self._build_ui()

    def _on_filepick(self, filename: str, dirpath: str):
        if dirpath:
            path = posixpath.join(dirpath, filename)
            if utils.path.exists(path):
                self._filepicker.hide()
                save_to_log(CACHE, {"root_dir": dirpath, "root_file": filename})
                self._ui_root_dir.set_value(path)

    def _on_outpick(self, path: str):
        self._outpicker.hide()
        save_to_log(CACHE, {"out_dir": path})
        self._ui_out_dir.set_value(path)

    def _on_load_config(self, filename: str, dirpath: str):
        self._configpicker.hide()
        path = posixpath.join(dirpath, filename)
        assert re.search("^.*\.(usd|usda|usdc|USD|USDA|USDC)$", path)  # Confirm path is a valid USD
        assert utils.path.exists(path)  # Ensure path exists
        save_to_log(CACHE, {"config_dir": dirpath})
        if path not in self.presets:
            self.presets.append(path)
            self._preset_model.append_child_item(None, ui.SimpleStringModel(posixpath.splitext(filename)[0]))
        self._preset_model.get_item_value_model().set_value(self.presets.index(path))

    def _on_load_json(self, path: str):
        self._jsonpicker.hide()
        assert re.search("^.*\.(json)$", path)  # Confirm path is a valid json file
        assert utils.path.exists(path)  # Ensure path exists
        save_to_log(CACHE, {"json_dir": posixpath.dirname(path)})
        with open(path, "r") as f:
            data = json.load(f)
        return data

    async def _on_root_dir_changed(self, path: str):
        """
        root usd directory changed
        """
        if utils.path.exists(path):
            self._settings.set("/kaolin/mode", 2)  # Set app in data generation mode
            self._reset()
            self._settings.set("/app/asyncRendering", False)  # Necessary to ensure correct GT output
            self._settings.set("/app/hydraEngine/waitIdle", True)  # Necessary to ensure correct GT output
            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            vis_prim = stage.GetPrimAtPath(SCENE_PATH)
            if vis_prim and self._preset_layer is None:
                omni.kit.commands.execute("DeletePrimsCommand", paths=[vis_prim.GetPath()])
            elif vis_prim and stage.GetPrimAtPath(f"{vis_prim.GetPath()}/Asset/Rig"):
                rig = stage.GetPrimAtPath(f"{vis_prim.GetPath()}/Asset/Rig")
                for child in rig.GetChildren():
                    self._set_visible(str(child.GetPath()), False)
            self.root_dir = path
            self.asset_list = await utils.path.get_usd_files_async(self.root_dir)
            if not self.option_frame:
                self._build_ui()
                if self.option_frame:
                    self.option_frame.visible = True
            await self.preview()
            self._preview_window.visible = False
        else:
            carb.log_error(f"[kaolin_app.research.data_generator] Directory not found: '{path}'")

    def _set_settings(self, width: int, height: int, renderer: str, **kwargs):
        self._settings.set("/app/renderer/resolution/width", width)
        self._settings.set("/app/renderer/resolution/height", height)
        self._settings.set("/rtx/rendermode", renderer)
        self._settings.set("/app/viewport/grid/enabled", False)
        self._settings.set("/app/viewport/grid/showOrigin", False)

    def _on_save_config(self, filename: str, dirname: str):
        assert utils.path.exists(dirname)
        self._configsaver.hide()

        # add sensor config to main config
        self.config["sensors"] = {s: True for s, v in self.sdv._sensors["Viewport"].items() if v["enabled"]}

        save_to_log(CACHE, {"config_dir": dirname})

        if self._preset_layer is None:
            raise ValueError("Something went wrong, Unable to save config.")

        # Create new layer
        filename = f"{posixpath.splitext(filename)[0]}.usda"
        new_path = posixpath.join(dirname, filename)
        if Sdf.Find(new_path) == self._preset_layer:
            new_layer = self._preset_layer
        else:
            # Transfer layer content over to new layer
            new_layer = Sdf.Layer.CreateNew(new_path)
            new_layer.TransferContent(self._preset_layer)
        new_layer.customLayerData = {"DataGenerator": self.config}
        new_layer.Save()

        self._on_load_config(filename, dirname)

    def _on_resolution_changed(self, model, option):
        # TODO type
        value = model.as_int
        self.config.update({option: value})
        self._settings.set(f"/app/renderer/resolution/{option}", value)
        model.set_value(value)

    def _on_preset_changed(self, path: str, update_config: bool = True) -> None:
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer()
        if self._preset_layer is not None:
            delete_sublayer(self._preset_layer)
        vis_prim = stage.GetPrimAtPath(SCENE_PATH)
        if vis_prim:
            omni.kit.commands.execute("DeletePrimsCommand", paths=[vis_prim.GetPath()])

        omni.kit.commands.execute(
            "CreateSublayerCommand",
            layer_identifier=root_layer.identifier,
            sublayer_position=-1,
            new_layer_path=path,
            transfer_root_content=False,
            create_or_insert=False,
        )

        self._preset_layer = Sdf.Find(root_layer.subLayerPaths[-1])
        if update_config:
            config = self._preset_layer.customLayerData.get("DataGenerator")
            if config:
                self.config = config

        if "sensors" in self.config:
            # Enable sensors
            for s in self.config["sensors"]:
                self.sdv._sensors["Viewport"][s]["enabled"] = True

        # Set preset as authoring layer
        edit_target = Usd.EditTarget(self._preset_layer)
        stage = omni.usd.get_context().get_stage()
        if not stage.IsLayerMuted(self._preset_layer.identifier):
            stage.SetEditTarget(edit_target)

        self.dr_components = {}
        for prim in stage.Traverse():
            if str(prim.GetTypeName()) in DR_COMPONENTS:
                key = prim.GetParent().GetName()
                self.dr_components.setdefault(key, []).append(prim)

        self.camera = stage.GetPrimAtPath(f"{SCENE_PATH}/CameraRig/Boom/Camera")
        self.create_asset_prim()
        self.option_frame.clear()
        with self.option_frame:
            self._build_ui_options()

    async def _preview_trajectory(self):
        stage = omni.usd.get_context().get_stage()
        trajectory_viz = stage.GetPrimAtPath(f"{SCENE_PATH}/Viz")
        if not trajectory_viz:
            carb.log_warn("Unable to preview trajectory, no trajectory detected.")
            return

        trajectory_viz.GetAttribute("visibility").Set("inherited")
        viewport = omni.kit.viewport.get_viewport_interface()
        omni.usd.get_context().get_selection().set_selected_prim_paths([f"{SCENE_PATH}/Viz"], True)
        await omni.kit.app.get_app_interface().next_update_async()
        viewport.get_viewport_window().focus_on_selected()
        omni.usd.get_context().get_selection().clear_selected_prim_paths()

    def _set_trajecotry_preview_visibility(self):
        show_preview = (
            self.config.get("cameramode") == "Trajectory" and self.config.get("trajectory_mode") == "CustomJson"
        )
        self._set_visible(f"{SCENE_PATH}/Viz", show_preview)

    def _on_trajectory_mode_changed(self, trajectory_mode_model):
        trajectory_mode = TRAJ_OPTIONS[trajectory_mode_model.get_item_value_model().as_int]
        self.config.update({"trajectorymode": trajectory_mode})
        self._set_trajecotry_preview_visibility()

    def _ui_modal(self, title: str, text: str, no_close: bool = False, ok_btn: bool = True):
        """ Create a modal window. """
        window_flags = ui.WINDOW_FLAGS_NO_RESIZE
        window_flags |= ui.WINDOW_FLAGS_NO_SCROLLBAR
        window_flags |= ui.WINDOW_FLAGS_MODAL
        if no_close:
            window_flags |= ui.WINDOW_FLAGS_NO_CLOSE
        modal = ui.Window(title, width=400, height=100, flags=window_flags)
        with modal.frame:
            with ui.VStack(spacing=5):
                text = ui.Label(text, word_wrap=True, style={"alignment": ui.Alignment.CENTER})
                if ok_btn:
                    btn = ui.Button("OK")
                    btn.set_clicked_fn(lambda: self._ui_toggle_visible([modal]))
        return modal

    def _ui_create_xyz(self, option, value=(0, 0, 0), idx=None, dtype=float):
        # TODO type
        colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
        with ui.HStack():
            for i, (label, colour) in enumerate(colors.items()):
                if i != 0:
                    ui.Spacer(width=4)
                with ui.ZStack(height=14):
                    with ui.ZStack(width=16):
                        ui.Rectangle(name="vector_label", style={"background_color": colour, "border_radius": 3})
                        ui.Label(label, alignment=ui.Alignment.CENTER)
                    with ui.HStack():
                        ui.Spacer(width=14)
                        self._ui_create_value(option, value[i], idx_opt=idx, idx=i, dtype=dtype)
                ui.Spacer(width=4)

    def _ui_create_value(self, option, value=0.0, idx=None, idx_opt=None, dtype=float):
        # TODO type
        if dtype == int:
            widget = ui.IntDrag(min=0, max=int(1e6))
        elif dtype == float:
            widget = ui.FloatDrag(min=-1e6, max=1e6, step=0.1, style={"border_radius": 1})
        elif dtype == bool:
            widget = ui.CheckBox()
        else:
            raise NotImplementedError
        widget.model.set_value(value)
        widget.model.add_value_changed_fn(
            lambda m: self._on_value_changed(option, m.get_value_as_float(), idx=idx, idx_opt=idx_opt)
        )
        # widget.model.add_value_changed_fn(lambda _: asyncio.ensure_future(self.render_asset())
        return widget

    def _ui_simple_block(self, label, option, is_xyz=False, dtype=float):
        # TODO type
        ui_fn = self._ui_create_xyz if is_xyz else self._ui_create_value
        with ui.HStack(spacing=5):
            ui.Label(label, width=120, height=10)
            ui_fn(option, value=self.config[option], dtype=dtype)

    def _ui_option_block(self, label, option, is_xyz=False, dtype=float):
        """
        Create option block on the UI
        """
        if option not in self.config:
            return None
        ui_fn = self._ui_create_xyz if is_xyz else self._ui_create_value
        option_block = ui.HStack(spacing=5)
        with option_block:
            ui.Label(label, width=120, height=10)
            model = ui.ComboBox(self.config[option]["mode"], "Fixed", "Random", width=80).model
            # create option based on "fixed" or "random"
            option_0 = ui.HStack(spacing=5)  # fixed
            option_1 = ui.VStack(spacing=5)  # random
            with option_0:
                ui_fn(option, value=self.config[option]["fixed"], dtype=dtype)
            with option_1:
                for i, m in enumerate(["Min", "Max"]):
                    with ui.HStack(spacing=5):
                        ui.Label(m, width=30)
                        ui_fn(option, value=self.config[option]["random"][i], idx=i, dtype=dtype)
            if self.config[option]["mode"] == 0:
                option_1.visible = False
            else:
                option_0.visible = False
            model.add_item_changed_fn(lambda m, i: self._ui_toggle_visible([option_0, option_1]))
            model.add_item_changed_fn(
                lambda m, i: self.config[option].update({"mode": m.get_item_value_model().as_int})
            )
        return option_block

    def _ui_toggle_visible(self, ui_elements):
        # TODO type
        for ui_el in ui_elements:
            ui_el.visible = not ui_el.visible

    def _build_run_ui(self):
        with self._window.frame:
            pass

    def _ui_up_axis(self):
        collection = ui.RadioCollection()
        with ui.HStack():
            ui.Label("Up Axis", width=120)
            with ui.HStack():
                ui.RadioButton(text="Y", radio_collection=collection, height=30)
                ui.RadioButton(text="Z", radio_collection=collection, height=30)
                collection.model.add_value_changed_fn(self._change_up_axis)
                collection.model.set_value(self.config.get("up_axis", 0))

    def _build_ui(self):
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=5):
                    with ui.HStack(spacing=5, height=15):
                        ui.Label("Root Dir", width=55)
                        self._ui_root_dir = ui.StringField().model
                        if self.root_dir:
                            self._ui_root_dir.set_value(self.root_dir)
                        self._ui_root_dir.add_value_changed_fn(
                            lambda m: asyncio.ensure_future(self._on_root_dir_changed(m.as_string))
                        )
                        browse = ui.Button(
                            image_url="resources/icons/folder.png",
                            width=30,
                            height=25,
                            style={"Button": {"margin": 0, "padding": 5, "alignment": ui.Alignment.CENTER}},
                        )
                        browse.set_clicked_fn(
                            lambda f=self._filepicker: self._show_filepicker(f, self._cache.get("root_dir", ""))
                        )

                    if self.root_dir:
                        with ui.HStack(height=0):
                            ui.Label("Presets", width=60)
                            self._preset_model = ui.ComboBox(
                                0, *[posixpath.splitext(posixpath.basename(p))[0] for p in self.presets]
                            ).model
                            config_dir = self._cache.get("config_dir", "")
                            config_file = self._cache.get("config_file", "")
                            ui.Button(
                                "Save As...",
                                clicked_fn=lambda f=self._configsaver: self._show_filepicker(
                                    f, config_dir, config_file
                                ),
                            )
                            ui.Button(
                                "Import",
                                clicked_fn=lambda f=self._configpicker: self._show_filepicker(
                                    f, config_dir, config_file
                                ),
                            )
                        self.option_frame = ui.VStack(spacing=5)
                        self.option_frame.visible = False
                        self._preset_model.add_item_changed_fn(
                            lambda m, i: self._on_preset_changed(self.presets[m.get_item_value_model().as_int])
                        )
                        if self.presets and not self._preset_layer:
                            self._on_preset_changed(self.presets[0])
                        self._build_progress_ui()
                    ui.Spacer()
                    ui.Button("Demo", clicked_fn=lambda: webbrowser.open(DEMO_URL), height=60)

    def _build_ui_options(self):
        # Output
        with ui.CollapsableFrame(title="Output", height=10):
            with ui.VStack(spacing=5):
                with ui.HStack(spacing=5, height=10):
                    ui.Label(
                        "Output Dir",
                        width=120,
                        height=10,
                        tooltip="Select directory to save output to. Existing files of the same name will be overwritten.",
                    )
                    self._ui_out_dir = ui.StringField().model
                    self._ui_out_dir.set_value(self.config["out_dir"])
                    self._ui_out_dir.add_value_changed_fn(lambda m: self.config.update({"out_dir": m.as_string}))
                    browse = ui.Button(
                        image_url="resources/icons/folder.png",
                        width=30,
                        height=25,
                        style={"Button": {"margin": 0, "padding": 5, "alignment": ui.Alignment.CENTER}},
                    )
                    browse.set_clicked_fn(
                        lambda f=self._outpicker: self._show_filepicker(f, self._cache.get("out_dir", ""))
                    )
                with ui.HStack(spacing=5, height=10):
                    ui.Label(
                        "Renders per Scene",
                        width=120,
                        height=10,
                        tooltip="Number of randomized scenes to be captured before re-sampling a new scene.",
                    )
                    model = ui.IntDrag(min=1, max=int(1e6)).model
                    model.set_value(self.config["renders_per_asset"])
                    model.add_value_changed_fn(
                        lambda m: self.config.update({"renders_per_asset": m.get_value_as_int()})
                    )
                _build_ui_sensor_selection("Viewport")
        # Assets
        with ui.CollapsableFrame(title="Assets", height=10):
            with ui.VStack(spacing=5):
                self._ui_simple_block("Fix Bottom Elevation", "asset_override_bottom_elev", dtype=bool)
                self._ui_simple_block("Normalize", "asset_normalize", dtype=bool)
                self._ui_up_axis()
                ui.Spacer()
        # Camera
        with ui.CollapsableFrame(title="Camera", height=10):
            with ui.VStack(spacing=5):
                with ui.HStack(spacing=5):
                    ui.Label(
                        "Camera Mode",
                        width=120,
                        height=10,
                        tooltip="Select random camera poses or follow a trajectory.",
                    )

                    cur_camera_idx = CAMERAS.index(self.config.get("cameramode", "UniformSampling"))
                    camera_mode_model = ui.ComboBox(cur_camera_idx, *CAMERAS, width=150).model
                    camera_mode_model.add_item_changed_fn(
                        lambda m, i: self.config.update({"cameramode": CAMERAS[m.get_item_value_model().as_int]})
                    )

                if "camera_focal_length" not in self.config:
                    self.config["camera_focal_length"] = {"fixed": 24.0, "mode": 0, "random": Gf.Vec2f([1.0, 120.0])}

                uniform_options = [
                    self._ui_option_block("Focal Length", "camera_focal_length"),
                    self._ui_option_block("Look-at Position", "centre", is_xyz=True),
                    self._ui_option_block("Distance", "distance"),
                    self._ui_option_block("Elevation", "elevation"),
                    self._ui_option_block("Azimuth", "azimuth"),
                ]

                if cur_camera_idx == 1:
                    self._ui_toggle_visible(uniform_options)
                camera_mode_model.add_item_changed_fn(lambda m, i: self._ui_toggle_visible(uniform_options))
                camera_mode_model.add_item_changed_fn(lambda *_: self._set_trajecotry_preview_visibility())

                # an indicator on turning on the trajectory
                traject_block = ui.VStack(spacing=5)
                with traject_block:
                    with ui.HStack(spacing=5):
                        ui.Label("Trajectory Mode", width=120, height=10, tooltip="Trajectory mode")
                        if "trajectorymode" not in self.config:
                            self.config["trajectorymode"] = "Spiral"
                        cur_traj_idx = TRAJ_OPTIONS.index(self.config.get("trajectorymode", "Spiral"))
                        trajmodel = ui.ComboBox(cur_traj_idx, *TRAJ_OPTIONS, width=150).model
                        trajmodel.add_item_changed_fn(lambda m, _: self._on_trajectory_mode_changed(m))
                    # spiral option
                    spiral_block = ui.VStack(spacing=5)
                    with spiral_block:
                        self._ui_option_block("Distance", "distance")  # distance block
                        with ui.HStack(spacing=5):  # elevation range block
                            ui.Label("Elevation Range", width=120, height=10, tooltip="Elevation range two numbers")
                            ui.Spacer(width=10)
                            for i, m in enumerate(["Min", "Max"]):
                                with ui.HStack(spacing=5):
                                    ui.Label(m, width=30)
                                    val = self.config["elevation"]["random"]
                                    self._ui_create_value("elevation", value=val[i], idx=i, dtype=float)
                        with ui.HStack(spacing=5):  # rotation block
                            ui.Label("Number of Rotations", width=120, height=10)
                            self.config["num_rotations"] = 3
                            n_rot = self.config.get("num_rotations")
                            self._ui_create_value("num_rotations", value=n_rot, dtype=int)
                        ui.Spacer()
                    spiral_block.visible = cur_traj_idx == 0
                    trajmodel.add_item_changed_fn(lambda m, i: self._ui_toggle_visible([spiral_block]))

                    # jsonoption
                    json_block = ui.VStack(spacing=5)
                    with json_block:
                        with ui.HStack(spacing=5, height=15):
                            ui.Label("Json path", width=55)
                            ui.Button(
                                "Json File",
                                clicked_fn=lambda f=self._jsonpicker: self._show_filepicker(
                                    f, self._cache.get("json_dir", "")
                                ),
                            )
                            if self.config.get("jsonpath") and os.path.exists(self.config["jsonpath"]):
                                asyncio.ensure_future(self._import_trajectory_from_json(self.config["jsonpath"]))
                        ui.Button(
                            "View Trajectory", clicked_fn=lambda: asyncio.ensure_future(self._preview_trajectory())
                        )
                        ui.Spacer()
                    json_block.visible = cur_traj_idx == 1
                    trajmodel.add_item_changed_fn(lambda m, i: self._ui_toggle_visible([json_block]))

                traject_block.visible = cur_camera_idx == 1
                camera_mode_model.add_item_changed_fn(lambda m, i: self._ui_toggle_visible([traject_block]))
                ui.Spacer()
                ui.Spacer()
        # Create UI elements for DR Components
        for title, components in self.dr_components.items():
            build_component_frame(title, components)

        # Render
        with ui.CollapsableFrame(title="Render Settings", height=10):
            self._settings.set("/rtx/rendermode", self.config["renderer"])
            self._settings.set("/rtx/pathtracing/totalSpp", self.config["spp"])
            self._settings.set("/rtx/pathtracing/optixDenoiser/enabled", self.config["denoiser"])
            self._settings.set("/rtx/pathtracing/clampSpp", 0)  # Disable spp clamping
            self._settings.set("/rtx/post/aa/op", 2)

            with ui.VStack(spacing=5):
                with ui.HStack(spacing=5):
                    ui.Label("Resolution", width=120)
                    ui.Label("Width", width=40, tooltip="Rendered resolution width, in pixels.")
                    width = ui.IntDrag(min=MIN_RESOLUTION["width"], max=MAX_RESOLUTION["width"]).model
                    width.add_value_changed_fn(lambda m: self._on_resolution_changed(m, "width"))
                    ui.Spacer(width=10)
                    ui.Label("Height", width=40, tooltip="Rendered resolution height, in pixels.")
                    height = ui.IntDrag(min=MIN_RESOLUTION["height"], max=MAX_RESOLUTION["height"]).model
                    height.add_value_changed_fn(lambda m: self._on_resolution_changed(m, "height"))
                    width.set_value(self.config.get("width", self._settings.get("/app/renderer/resolution/width")))
                    height.set_value(self.config.get("height", self._settings.get("/app/renderer/resolution/height")))

                with ui.HStack(spacing=5):
                    ui.Label("Renderer", width=120, tooltip="Render Mode")
                    cur_renderer_idx = RENDERERS.index(self.config["renderer"])
                    model = ui.ComboBox(cur_renderer_idx, *RENDERERS, width=200).model
                    model.add_item_changed_fn(
                        lambda m, i: self.config.update({"renderer": RENDERERS[m.get_item_value_model().as_int]})
                    )
                    model.add_item_changed_fn(
                        lambda m, i: self._settings.set("/rtx/rendermode", RENDERERS[m.get_item_value_model().as_int])
                    )
                pt_block = ui.VStack(spacing=5)
                with pt_block:
                    with ui.HStack(spacing=5):
                        ui.Label(
                            "Samples Per Pixel", width=120, tooltip="Number of samples taken at each pixel, per frame."
                        )
                        spp = ui.IntDrag().model
                        spp.set_value(self.config["spp"])
                        spp.add_value_changed_fn(
                            lambda m: self.config.update({"spp": m.as_int})
                        )  # Only change SPP during run
                        spp.add_value_changed_fn(
                            lambda m: self._settings.set("/rtx/pathtracing/totalSpp", m.as_int)
                        )  # SPP Max
                    with ui.HStack(spacing=5):
                        ui.Label("Denoiser", width=120, tooltip="Toggle denoiser")
                        denoiser = ui.CheckBox().model
                        denoiser.set_value(self.config["denoiser"])
                        denoiser.add_value_changed_fn(lambda m: self.config.update({"denoiser": m.as_bool}))
                        denoiser.add_value_changed_fn(
                            lambda m: self._settings.set("/rtx/pathtracing/optixDenoiser/enabled", m.as_bool)
                        )
                    ui.Spacer()
                pt_block.visible = bool(cur_renderer_idx)
                model.add_item_changed_fn(lambda m, i: self._ui_toggle_visible([pt_block]))
                with ui.HStack():
                    ui.Label("Subdiv", width=120, tooltip="Subdivision Global Refinement Level")
                    with ui.HStack():
                        ui.Label("Refinement Level", width=100, tooltip="Subdivision Global Refinement Level")
                        subdiv = ui.IntDrag(min=0, max=2).model
                        subdiv.add_value_changed_fn(lambda m: self.config.update({"subdiv": m.as_int}))
                        subdiv.add_value_changed_fn(
                            lambda m: self._settings.set("/rtx/hydra/subdivision/refinementLevel", m.as_int)
                        )
                ui.Spacer()

        with ui.HStack(spacing=5):
            btn = ui.Button("Preview", height=40, tooltip="Render a preview with the current settings.")
            btn.set_clicked_fn(lambda: asyncio.ensure_future(self.preview()))
            btn = ui.Button("Run", height=40, tooltip="Generate and save groundtruth with the current settings.")
            btn.set_clicked_fn(lambda: asyncio.ensure_future(self.run()))

    def _build_progress_ui(self):
        self.progress = {"block": ui.VStack(spacing=5), "stop_signal": False}
        self.progress["block"].visible = False
        with self.progress["block"]:
            with ui.HStack(height=0):
                ui.Label(
                    "TOTAL",
                    width=80,
                    style={"font_size": 20.0},
                    tooltip="Render progress of all scenes to be rendered.",
                )
                self.progress["bar1"] = ui.ProgressBar(height=40, style={"font_size": 20.0}).model
            with ui.HStack(height=0):
                ui.Label(
                    "Per Scene",
                    width=80,
                    style={"font_size": 16.0},
                    tooltip="Render progress of the total number of renders for this scenes",
                )
                self.progress["bar2"] = ui.ProgressBar(height=20, style={"font_size": 16.0}).model
            btn = ui.Button("Cancel", height=60)
            btn.set_clicked_fn(lambda: self.progress.update({"stop_signal": True}))

    @staticmethod
    def get_instance():
        return _extension_instance
