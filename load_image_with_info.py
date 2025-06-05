import hashlib
import os
import json
from PIL import Image, ImageOps, ImageSequence, ExifTags
from PIL.PngImagePlugin import PngInfo
import pillow_avif
import numpy as np
import torch
import folder_paths
import node_helpers

class LoadImageWithInfo:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        img_exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".avif", ".AVIF", ".tif", ".tiff", ".TIF", ".TIFF"]
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[1] in img_exts]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES =("IMAGE","MASK","STRING","STRING","INT","INT","INT","INT","INT","INT","STRING")
    RETURN_NAMES = ("image","mask","filename","format","dpi","width","height","long_edge","short_edge","file_size","exif")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        image_name, image_format = os.path.splitext(os.path.basename(image_path))
        image_format = image_format[1:] or 'png'
        image_file_size = os.path.getsize(image_path)

        img = node_helpers.pillow(Image.open, image_path)

        # 获取图像基本信息
        width, height = img.size
        long_edge = max(width, height)
        short_edge = min(width, height)
        
        # 获取DPI信息
        try:
            dpi = img.info.get('dpi', (96, 96))[0]
        except:
            dpi = 0
        
        # 获取EXIF信息
        exif_data = {}
        try:
            exif = {ExifTags.TAGS[k]: v for k, v in img.getexif().items() if k in ExifTags.TAGS} if img.getexif() else {}
            for key, value in exif.items():
                if isinstance(value, bytes):
                    try:
                        exif_data[key] = value.decode('utf-8')
                    except:
                        exif_data[key] = str(value)
                else:
                    exif_data[key] = str(value)
        except:
            pass

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (
            output_image,
            output_mask,
            image_name,
            image_format,
            dpi,
            width,
            height,
            long_edge,
            short_edge,
            image_file_size,
            exif_data
        )

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class SaveImageWithInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "image"}),
                "format": (["original", "avif", "webp", "jpg", "png", "tiff"], {"default": "original"}),
                "original_format": ("STRING", {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "display": "silder", 'tooltip': "Quality for JPEG/WebP/AVIF formats; Quality is relative to each format. \n* Example: AVIF 60 is same quality as WebP 90. \n* PNG compression is fixed at 4 and not affected by this. PNG compression times skyrocket above level 4 for zero benefits on filesize."}),
                "dpi": ("INT", {"default": 96}),
                "exif": ("STRING", {"default": "{}"}),
                'always_save_png': ('BOOLEAN', {'default': True, 'tooltip': "总是保存为PNG格式，即使选择了其他格式。"}),
                'image_preview': ('BOOLEAN', {'default': True, 'tooltip': "Turns the image preview on and off"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input image with to your ComfyUI output directory."

    type = 'output'
    quality = 95
    quality_avif = 80
    # optimize_image only works for jpeg, png and TIFF, with like just 2% reduction in size; not used for PNG as it forces a level 9 compression.
    optimize_image = True

    def save_image(self, image, filename, format, original_format, quality, dpi, exif, always_save_png, image_preview, prompt=None, extra_pnginfo=None):
        results = []

        # 确定保存格式
        save_format = original_format if format == "original" else format
        
        # 构建完整文件名
        if filename.endswith(f".{save_format}"):
            filename = filename[:-len(f".{save_format}")]
        full_filename = f"{filename}.{save_format}"
        full_filename_png = f"{filename}.png"
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        
        # 构建完整路径
        full_path = os.path.join(output_dir, full_filename)
        LoadImageWithInfoPNG_dir = os.path.join(output_dir, "LoadImageWithInfoPNG")
        full_path_png = os.path.join(LoadImageWithInfoPNG_dir, full_filename_png)
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        if always_save_png: os.makedirs(LoadImageWithInfoPNG_dir, exist_ok=True)

        # 如果文件已存在，更改文件名
        if os.path.exists(full_path):
            base, ext = os.path.splitext(full_filename)
            counter = 0
            while os.path.exists(full_path):
                counter += 1
                full_filename = f"{base}_{counter}{ext}"
                full_path = os.path.join(output_dir, full_filename)
            full_filename_png = f"{full_filename_png[:-4]}_{counter}.png"
            full_path_png = os.path.join(LoadImageWithInfoPNG_dir, full_filename_png)
        
        # 处理EXIF数据
        try:
            if exif and exif != "{}":
                if isinstance(exif, str):
                    exif_data = json.loads(exif)
                else:
                    exif_data = exif
            else:
                exif_data = {}
        except:
            exif_data = {}
        
        # 保存图像
        img = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        # 设置DPI
        if dpi > 0:
            img.info['dpi'] = (dpi, dpi)
        
        kwargs = dict()
        kwargs_png = {
            'compress_level': 4,
            'pnginfo': self.genMetadataPng(prompt, extra_pnginfo)
        }
        
        # 根据格式保存图像
        if save_format.lower() == 'png':
            kwargs = kwargs_png
        else:
            kwargs["optimize"] = self.optimize_image
            if save_format.lower() == 'avif':
                if quality == 100:
                    kwargs["lossless"] = True
                else:
                    if quality == 0 or quality > self.quality_avif:
                        quality = self.quality_avif
                    kwargs['quality'] = quality
            elif save_format.lower() == 'webp':
                if quality == 100:
                    kwargs["lossless"] = True
                else:
                    if quality == 0:
                        quality = self.quality
                    kwargs['quality'] = quality
            elif save_format.lower() != 'tiff':
                if quality == 0:
                    quality = self.quality
                kwargs['quality'] = quality

                if save_format.lower() in ['jpg', 'jpeg']:
                    # 添加EXIF数据
                    if exif_data:
                        exif_bytes = img.getexif()
                        for k, v in exif_data.items():
                            try:
                                # 尝试找到EXIF标签的数字ID
                                tag_id = None
                                for tag, tag_name in ExifTags.TAGS.items():
                                    if tag_name == k:
                                        tag_id = tag
                                        break
                                
                                if tag_id:
                                    exif_bytes[tag_id] = v
                                else:
                                    # 如果找不到标签ID，尝试直接使用键名
                                    exif_bytes[k] = v
                            except:
                                pass
                        kwargs['exif'] = exif_bytes
                    kwargs["subsampling"] = 0
                else:
                    # 默认保存为PNG
                    kwargs = kwargs_png
                    full_filename = full_filename_png
                    full_path = os.path.join(output_dir, full_filename)
        
        img.save(full_path, **kwargs)
        if always_save_png: img.save(full_path_png, **kwargs_png)
        
        if image_preview:
            results.append({
                'filename': full_filename,
                'path': full_path,
                'type': self.type
            })
            
        return { "ui": { "images": results } }
    
    def genMetadataPng(self, prompt, extra_pnginfo=None):
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text('prompt', json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        
        return metadata

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LoadImageWithInfo": LoadImageWithInfo,
    "SaveImageWithInfo": SaveImageWithInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithInfo": "Load Image With Info",
    "SaveImageWithInfo": "Save Image With Info",
}