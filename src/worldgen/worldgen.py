import torch 
import numpy as np
import cv2
from PIL import Image
from .pano_depth import build_depth_model, pred_pano_depth, pred_depth
from .pano_depth_dap import build_dap_depth_model, pred_pano_depth_dap
from .pano_seg import build_segment_model, seg_pano_fg
from .pano_gen import build_pano_gen_model, gen_pano_image, build_pano_fill_model, gen_pano_fill_image
from .pano_inpaint import build_inpaint_model, inpaint_image
from .utils.splat_utils import convert_rgbd_to_gs, SplatFile, mask_splat, merge_splats
from .utils.general_utils import map_image_to_pano, resize_img, depth_match
from typing import Optional

class WorldGen:
    def __init__(self,
            mode: str = 't2s',
            use_sharp: bool = False,
            inpaint_bg: bool = False,
            lora_path: str = None,
            resolution: int = 1600,
            device: torch.device = 'cuda',
            low_vram: Optional[bool] = None,
            depth_backend: str = 'unik3d',
            dap_weights_path: Optional[str] = None,
            dap_depth_scale: float = 100.0,
            dap_num_passes: int = 2,
        ):
        self.device = device
        self.depth_backend = depth_backend
        self.dap_depth_scale = dap_depth_scale
        self.dap_num_passes = dap_num_passes
        self.mode = mode
        self.resolution = resolution

        if depth_backend == 'unik3d':
            self.depth_model = build_depth_model(device)
        elif depth_backend == 'dap':
            if dap_weights_path is None:
                raise ValueError("dap_weights_path must be provided when depth_backend='dap'")
            self.dap_model = build_dap_depth_model(dap_weights_path, device)
            if mode == 'i2s':
                # UniK3D still needed for perspective depth in i2s mode
                self.depth_model = build_depth_model(device)
        else:
            raise ValueError(f"Unknown depth_backend: '{depth_backend}'. Use 'unik3d' or 'dap'")

        # Set low_vram based on available VRAM if not specified
        if low_vram is None:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            low_vram = total_vram < 24
            print(f"Detected {total_vram:.1f}GB VRAM, {'enabling' if low_vram else 'disabling'} low VRAM mode")
        self.low_vram = low_vram

        if mode == 't2s':
            self.pano_gen_model = build_pano_gen_model(lora_path=lora_path, device=device, low_vram=low_vram)
        elif mode == 'i2s':
            self.pano_gen_model = build_pano_fill_model(lora_path=lora_path, device=device, low_vram=low_vram)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode must be 't2s' or 'i2s'")

        self.use_sharp = use_sharp
        if use_sharp:
            from .pano_sharp import build_sharp_model, predict_equirectangular
            self.sharp_model = build_sharp_model(device=device)

        self.inpaint_bg = inpaint_bg
        if inpaint_bg:
            self.seg_processor, self.seg_model = build_segment_model(device)
            self.inpaint_pipe = build_inpaint_model(device)

    def depth2gs(self, predictions) -> SplatFile:
        rgb = predictions["rgb"]
        distance = predictions["distance"]
        rays = predictions["rays"]
        splat = convert_rgbd_to_gs(rgb, distance, rays)
        return splat
    
    def inpaint_bg_splat(self, pano_image: Image.Image, init_splat: SplatFile, init_pred: dict) -> SplatFile:
        fg_mask = seg_pano_fg(self.seg_processor, self.seg_model, pano_image, init_pred["distance"])
        edge_mask = cv2.dilate(fg_mask, np.ones((3,3), np.uint8), iterations=1) - cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        init_splat = mask_splat(init_splat, (1-edge_mask))
        
        dilated_fg_mask = cv2.dilate(fg_mask, np.ones((5,5), np.uint8), iterations=10)
        pano_bg = inpaint_image(self.inpaint_pipe, pano_image, dilated_fg_mask)
        bg_pred = self._pred_pano_depth(pano_bg)
        bg_pred = depth_match(init_pred, bg_pred, (1-dilated_fg_mask))
        pano_bg_splat = self.depth2gs(bg_pred)
        occ_bg_splat = mask_splat(pano_bg_splat, dilated_fg_mask)
        merged_splat = merge_splats(init_splat, occ_bg_splat)
        return merged_splat
    
    def _pred_pano_depth(self, pano_image: Image.Image) -> dict:
        if self.depth_backend == 'dap':
            return pred_pano_depth_dap(self.dap_model, pano_image, self.dap_depth_scale, self.dap_num_passes)
        return pred_pano_depth(self.depth_model, pano_image)

    def _generate_world(self, pano_image: Image.Image) -> SplatFile:
        init_pred = self._pred_pano_depth(pano_image)

        splat = self.depth2gs(init_pred)
        if self.inpaint_bg:
            splat = self.inpaint_bg_splat(pano_image, splat, init_pred)
        return splat

    
    def generate_pano(self, prompt: str = "", image: Optional[Image.Image] = None) -> Image.Image:
        if self.mode == 't2s':
            assert image is None, "image is not supported for text-to-scene generation"
            pano_image = gen_pano_image(self.pano_gen_model, prompt=prompt, height=self.resolution//2, width=self.resolution)
        elif self.mode == 'i2s':
            assert image is not None, "image is required for image-to-scene generation"
            image = resize_img(image) # Limit the longest edge to 1024 to avoid OOM
            predictions = pred_depth(self.depth_model, image)
            pano_cond_img, cond_mask = map_image_to_pano(predictions, device=self.device)
            pano_image = gen_pano_fill_image(
                self.pano_gen_model, 
                image=pano_cond_img, 
                mask=cond_mask,
                prompt=prompt, 
                height=self.resolution//2, 
                width=self.resolution
            )

            # Remap original image to pano image with higher resolution
            map_height, map_width = pano_cond_img.height, pano_cond_img.width
            pano_image = pano_image.resize((map_width, map_height))
            pano_cond_img, mask = np.array(pano_cond_img), np.array(cond_mask) / 255.0
            pano_image = np.array(pano_image) * mask[:,:,None] + pano_cond_img * (1-mask[:,:,None])
            pano_image = Image.fromarray(pano_image.astype(np.uint8))
        else:
            raise ValueError(f"Invalid mode: {self.mode}, mode must be 't2s' or 'i2s'")
        return pano_image
    
    @torch.inference_mode()
    def generate_world(
        self, 
        prompt: str = "", 
        image: Optional[Image.Image] = None
    ) -> SplatFile:
        pano_image = self.generate_pano(prompt, image)
        scene = self._generate_world(pano_image)
        return scene