import hashlib
import os
from PIL import Image, ImageOps, ImageSequence, ExifTags
import pillow_avif
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
            dpi = 96
        
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
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "image"}),
                "format": (["original", "avif", "webp", "jpg", "png"], {"default": "original"}),
                "original_format": ("STRING", {"default": "png"}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "dpi": ("INT", {"default": 96}),
                "exif": ("STRING", {"default": "{}"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filename")
    FUNCTION = "save_image"
    OUTPUT_NODE = True

    def save_image(self, images, filename, format, original_format, quality, dpi, exif, prompt=None, extra_pnginfo=None):
        # 确定保存格式
        save_format = original_format if format == "original" else format
        
        # 构建完整文件名
        full_filename = f"{filename}.{save_format}" if not filename.endswith(f".{save_format}") else filename
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        
        # 构建完整路径
        full_path = os.path.join(output_dir, full_filename)
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理EXIF数据
        try:
            if exif and exif != "{}":
                if isinstance(exif, str):
                    import json
                    exif_data = json.loads(exif)
                else:
                    exif_data = exif
            else:
                exif_data = {}
        except:
            exif_data = {}
        
        # 保存图像
        results = []
        for i, image in enumerate(images):
            img = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            
            # 设置DPI
            if dpi > 0:
                img.info['dpi'] = (dpi, dpi)
            
            # 构建单个图像的文件名
            if i > 0:
                current_filename = f"{filename}_{i}.{save_format}"
                current_path = os.path.join(output_dir, current_filename)
            else:
                current_filename = full_filename
                current_path = full_path
            
            # 根据格式保存图像
            if save_format.lower() == 'png':
                # PNG的compress_level范围是0-9，将quality(1-100)映射到compress_level(0-9)
                # 注意：对于PNG，较低的compress_level意味着较低的压缩率和较高的质量
                compress_level = max(0, min(9, 9 - int(quality / 11)))
                img.save(current_path, format='PNG', compress_level=compress_level, pnginfo=None)
            elif save_format.lower() in ['jpg', 'jpeg']:
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
                    img.save(current_path, format='JPEG', quality=quality, exif=exif_bytes)
                img.save(current_path, format='JPEG', quality=quality)
            elif save_format.lower() == 'webp':
                img.save(current_path, format='WEBP', quality=quality)
            elif save_format.lower() == 'avif':
                img.save(current_path, format='AVIF', quality=quality)
            else:
                # 尝试使用原始格式保存
                try:
                    # 尝试使用quality参数，如果格式不支持则忽略
                    try:
                        img.save(current_path, format=save_format.upper(), quality=quality)
                    except TypeError:
                        img.save(current_path, format=save_format.upper())
                except:
                    # 如果失败，默认保存为PNG
                    current_filename = f"{filename}_{i if i > 0 else ''}.png"
                    current_path = os.path.join(output_dir, current_filename)
                    compress_level = max(0, min(9, 9 - int(quality / 11)))
                    img.save(current_path, format='PNG', compress_level=compress_level)
            
            results.append({
                'filename': current_filename,
                'path': current_path
            })
        
        # 添加到ComfyUI的保存图像列表中
        for result in results:
            folder_paths.add_to_output_list(result['path'])
        
        return (images, full_filename)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LoadImageWithInfo": LoadImageWithInfo,
    "SaveImageWithInfo": SaveImageWithInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithInfo": "Load Image With Info",
    "SaveImageWithInfo": "Save Image With Info",
}