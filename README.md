# ComfyUI LoadImageWithInfo 节点

这是一个用于ComfyUI的自定义节点，可以获取图像的详细信息，包括名称、格式（后缀）、DPI、尺寸、长边、短边、体积大小和EXIF数据。

## 功能

该节点可以提取以下图像信息：

- 文件名称（不带后缀）
- 文件格式（无点后缀）
- DPI
- 宽度和高度
- 长边和短边长度
- 文件大小（字节）
- EXIF元数据（如果存在）

## 安装

1. 将此仓库克隆或下载到ComfyUI的`custom_nodes`目录中：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sjh00/ComfyUI-LoadImageWithInfo.git
```

或者直接下载ZIP文件并解压到`custom_nodes`目录。

2. 重启ComfyUI

## 使用方法

1. 在ComfyUI工作流中，找到`image`分类下的`Load Image With Info`节点
2. 将图像连接到节点的`image`输入
3. 设置`filename`参数（可选，用于确定文件格式）
4. 节点将输出图像信息

## 示例输出

```json
{
  "filename": "example.jpg",
  "format": "jpg",
  "dpi": 72,
  "dimensions": "1920x1080",
  "width": 1920,
  "height": 1080,
  "long_edge": 1920,
  "short_edge": 1080,
  "file_size": "2.34 MB",
  "file_size_bytes": 2453664,
  "exif": {
    "Make": "Canon",
    "Model": "Canon EOS 5D Mark IV",
    "DateTime": "2023:01:15 14:30:22"
    // 更多EXIF数据...
  }
}
```

## 注意事项

- EXIF数据的可用性取决于图像是否包含这些元数据
- 对于ComfyUI生成的图像，某些信息（如DPI和EXIF）可能不存在或为默认值
- 文件大小是基于内存中的图像估算的，可能与实际保存到磁盘的文件大小略有不同

## 许可证

MIT