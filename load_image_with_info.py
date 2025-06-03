import hashlib
import os
import io
from PIL import Image, ImageOps, ImageSequence, ExifTags
import numpy as np
import torch
import folder_paths
import node_helpers

class LoadImageWithInfo:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "STRING", "INT", "INT", "INT", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image","mask","filename","format","dpi","width","height","long_edge","short_edge","file_size","exif")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        image_name = os.path.basename(image_path)
        image_format = os.path.splitext(image_path)[1][1:] or 'png'
        image_file_size = os.path.getsize(image_path)

        img = node_helpers.pillow(Image.open, image_path)

        # 获取图像基本信息
        width, height = img.size
        long_edge = max(width, height)
        short_edge = min(width, height)
        
        # 获取DPI信息
        try:
            dpi = img.info.get('dpi', (72, 72))[0]
        except:
            dpi = 72
        
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

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LoadImageWithInfo": LoadImageWithInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithInfo": "Load Image With Info",
}