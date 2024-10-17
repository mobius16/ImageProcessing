# 部署 teed、depth-anything
# 腐蚀算法
# 读取图片
# 输出图片
# 使用 depth-anything + teed 生成外轮廓
# 使用 teed + 腐蚀算法 生成内边缘
from PIL import Image

import cv2
import cv2_ext
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import TEED.main as teed
from TEED.main import parse_args

from depthAnything.depth_anything.dpt import DepthAnything
from depthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import shutil


def multiply_blend(image1, image2):
    # 将图片转换为浮点数，方便计算
    # Ensure image2 has the same shape as image1
    image2 = np.stack((image2,) * 3, axis=-1)
    # Perform the blending
    multiplied = np.multiply(image1 / 255.0, image2 / 255.0) * 255.0
    return multiplied.astype(np.uint8)

    # Example usage


image1 = np.random.randint(0, 256, (717, 790, 3), dtype=np.uint8)
image2 = np.random.randint(0, 256, (717, 790), dtype=np.uint8)

result = multiply_blend(image1, image2)
print(result.shape)  # Should be (717, 790, 3)

def screen_blend(image1, image2):
    # 将图片转换为浮点数，方便计算
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    # 执行滤色操作
    screened = 1 - (1 - image1 / 255) * (1 - image2 / 255) * 255

    # 将结果转换回uint8
    result = np.clip(screened, 0, 255).astype('uint8')
    return result

def erosion(img, kernel_size = 3, iterations = 1, dilate = False):
    
    # 灰度化
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # 二值化
    # _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # 腐蚀
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if dilate:
        img = cv2.dilate(img, kernel, iterations=iterations)
    else:
        img = cv2.erode(img, kernel, iterations=iterations)

    return img

def erosion_img_from_path(img_path, output_dir = './output/erosion_img', kernel_size = 3, iterations = 1, dilate = False):
    # 读取图片
    if os.path.isfile(img_path):
        name, extension = os.path.splitext(img_path)
        if extension:
            if extension.lower() == 'txt':
                with open(img_path, 'r',encoding= 'utf-8') as f:
                    filenames = f.read().splitlines()
            elif extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif']:
                filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.') and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif'))]
        filenames.sort()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(filenames):
        img = cv2.imread(filename)
        img = erosion(img, kernel_size, iterations, dilate)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), img)


def copy_file(src, dest):
    # 移动文件
    source = src
    destination = dest
    try:
        shutil.copy(source, destination)
    except IOError as e:
        print("Unable to copy file. %s" % e)


def guassian_blur_path(img_path, output_dir = './output/guassian_blur', kernel_size = 3, sigmaX = 0):
    # 读取图片
    if os.path.isfile(img_path):
        name, extension = os.path.splitext(img_path)
        if extension:
            if extension.lower() == 'txt':
                with open(img_path, 'r',encoding= 'utf-8') as f:
                    filenames = f.read().splitlines()
            elif extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif']:
                filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.') and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif'))]
        filenames.sort()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(filenames):
        img = cv2.imread(filename)
        img = cv2.GaussianBlur(img, (kernel_size,kernel_size), sigmaX)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), img)

def depth_anything(img_path = './input', outdir = './output/depth_anything', encoder = 'vitl', pred_only = True, grayscale = True):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img-path', type=str)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')
    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    # parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    # args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    depth_anything = DepthAnything(model_configs[encoder])
    depth_anything.load_state_dict(torch.load('./checkpoints/depth_anything_{}14.pth'.format(encoder)))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(img_path):
        name, extension = os.path.splitext(img_path)
        if extension:
            if extension.lower() == 'txt':
                with open(img_path, 'r',encoding= 'utf-8') as f:
                    filenames = f.read().splitlines()
            elif extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif']:
                filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.') and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif'))]
        filenames.sort()
    
    os.makedirs(outdir, exist_ok=True)
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        
        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        filename = os.path.basename(filename)
        
        if pred_only:
            cv2.imwrite(os.path.join(outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth])
            
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width
            
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            cv2.imwrite(os.path.join(outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)

def teed_imgs(img_path = './input', outdir = './output/teed_imgs',gaussianBlur = [0,3,0]):
    args, train_info = parse_args(is_testing=True, pl_opt_dir=outdir)
    os.makedirs('teed_tmp', exist_ok=True)
    if os.path.isfile(img_path):
        name, extension = os.path.splitext(img_path)
        if extension:
            if extension.lower() == 'txt':
                with open(img_path, 'r',encoding= 'utf-8') as f:
                    filenames = f.read().splitlines()
            elif extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif']:
                filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.') and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif'))]
        filenames.sort()
    for filename in tqdm(filenames):
        if gaussianBlur[0] != 0:
            img = cv2.imread(filename)
            img = cv2.GaussianBlur(img, (gaussianBlur[1],gaussianBlur[1]), gaussianBlur[2])
            cv2.imwrite(os.path.join('teed_tmp', os.path.basename(filename)), img)
        else:
            copy_file(filename, 'teed_tmp')
    teed.main(args, train_info)
    shutil.rmtree('teed_tmp')

def merge_2_images(img1, img2, mode, erosion_para = [[0,0],[0,0]], dilate = [0,0]): #将 img1 合并至 img2，调整大小与 img2 相同
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    if erosion_para[0][1] != 0:
        img1 = erosion(img1, erosion_para[0][0], erosion_para[0][1], dilate[0])
    if erosion_para[1][1] != 0:
        img2 = erosion(img2, erosion_para[1][0], erosion_para[1][1], dilate[1])
    if mode == 'multiply':
        return multiply_blend(img1, img2)
    elif mode == 'screen':
        return screen_blend(img1, img2)
    
def merge_images_in_2_folder(folder1, folder2, outdir, suffix_need_remove = None, suffix_floder = 0 , mode = 'multiply', erosion_para = [[0,0],[0,0]], dilate = [0,0]): #将 folder1 和 folder2 中的图片合并，可选是否移除某文件夹后缀，可选腐蚀参数[kernel_size,iterations]
    os.makedirs(outdir, exist_ok=True)
    name_extension_pairs_folder1 = [os.path.splitext(filename) for filename in os.listdir(folder1) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif'))]
    filenames_noext_folder1, extensions_folder1 = zip(*name_extension_pairs_folder1)
    name_extension_pairs_folder2 = [os.path.splitext(filename) for filename in os.listdir(folder2) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','tif'))]
    filenames_noext_folder2, extensions_folder2 = zip(*name_extension_pairs_folder2)
    if suffix_need_remove:
        if suffix_floder == 0:
            filenames_raw = list(filenames_noext_folder1).copy()
            filenames_noext_folder1 = [filename[:-len(suffix_need_remove)] + filename[-len(suffix_need_remove):].replace(suffix_need_remove, '') for filename in filenames_noext_folder1]
        if suffix_floder == 1:
            filenames_raw = list(filenames_noext_folder2).copy()
            filenames_noext_folder2 = [filename[:-len(suffix_need_remove)] + filename[-len(suffix_need_remove):].replace(suffix_need_remove, '') for filename in filenames_noext_folder2]

    for index, filename in enumerate(filenames_noext_folder1):
        if filename in filenames_noext_folder2:
            print(filename)
            if suffix_need_remove:
                if suffix_floder == 0:
                    img1 = os.path.join(folder1, filenames_raw[index] + extensions_folder1[index])
                    img2 = os.path.join(folder2, filename + extensions_folder2[filenames_noext_folder2.index(filename)])
                if suffix_floder == 1:
                    img1 = os.path.join(folder1, filename + extensions_folder1[index])
                    img2 = os.path.join(folder2, filenames_raw[filenames_noext_folder2.index(filename)] + extensions_folder2[filenames_noext_folder2.index(filename)])
            else:
                img1 = os.path.join(folder1, filename + extensions_folder1[index])
                img2 = os.path.join(folder2, filename + extensions_folder2[filenames_noext_folder2.index(filename)])
            result = merge_2_images(img1, img2, mode, erosion_para, dilate)
            cv2.imwrite(os.path.join(outdir, filename + extensions_folder1[index]), result)

def process_line(img_path = './input', outdir = './output'):
    depth_anything(img_path, os.path.join(outdir,"depth_anything"))
    teed_imgs(img_path, os.path.join(outdir,"teed_imgs"), [1,7,2])
    teed_imgs(os.path.join(outdir,"depth_anything"), os.path.join(outdir,"dp_teed_imgs"), [0,7,2])
    merge_images_in_2_folder(os.path.join(outdir,"teed_imgs"), os.path.join(outdir,"dp_teed_imgs"), os.path.join(outdir,"merged_imgs"),'_depth', 1, 'multiply', [[2,0],[2,1]],[1,0])

def invert_image(image):
    # 将图片从BGR转为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对灰度图进行反转
    inverted_image = cv2.bitwise_not(gray_image)
    # 将反转后的灰度图转换回BGR格式
    inverted_image_bgr = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)
    return inverted_image_bgr

def process_images(input_folder='./output/merged_imgs'):
    output_folder = os.path.join(os.path.dirname(input_folder), 'output_invert')
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 'tif'))]

    for image_file in tqdm(image_files):
        image_path = os.path.join(input_folder, image_file)
        try:
            # 使用PIL库读取图像
            with Image.open(image_path) as img:
                image = np.array(img.convert('RGB'))[:, :, ::-1].copy()
                if image is not None:
                    # 翻转图片
                    inverted_image = invert_image(image)
                    # 保存翻转后的图片到输出文件夹
                    output_path = os.path.join(output_folder, image_file)
                    cv2.imwrite(output_path, inverted_image)
                else:
                    raise ValueError(f"Failed to read image: {image_file}")
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

def process_line(img_path='./input', outdir='./output'):
    depth_anything(img_path, os.path.join(outdir, "depth_anything"))
    teed_imgs(img_path, os.path.join(outdir, "teed_imgs"), [1, 7, 2])
    teed_imgs(os.path.join(outdir, "depth_anything"), os.path.join(outdir, "dp_teed_imgs"), [0, 7, 2])
    merge_images_in_2_folder(os.path.join(outdir, "teed_imgs"), os.path.join(outdir, "dp_teed_imgs"), os.path.join(outdir, "merged_imgs"), '_depth', 1, 'multiply', [[2, 0], [2, 1]], [1, 0])
    process_images(os.path.join(outdir, "merged_imgs"))  # 处理merged_imgs文件夹中的图片




if __name__ == '__main__':
    # depth_anything()
    # teed_imgs('./input', './output/teed_imgs', [1,7,2])
    # teed_imgs('./output/depth_anything', './output/dp_teed_imgs', [0,7,2])
    # merge_images_in_2_folder('./output/teed_imgs', './output/dp_teed_imgs', './output/merged_imgs','_depth', 1, 'multiply', [[2,0],[2,1]],[1,0])

    # erosion_img_from_path('./output/teed_imgs', './output/erosion_imgs', 2, 1, True)
    # guassian_blur_path('./input', './output/guassian_blur', 7, 2)
    # erosion_img_from_path('./output/merged_imgs', './output/erosion_merged_imgs', 2, 1, False)
    # erosion_img_from_path('./output/erosion_merged_imgs', './output/erosion2_merged_imgs', 2, 1, True)
    img_path = "input"
    outdir = "output"
    process_line(img_path,outdir)
