import argparse
import torch
import os
import sys
import importlib
import numpy as np
from PIL import Image

# ---------------------- 设置路径 ----------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 只加根目录，不加子目录

print("Python sys.path 前几项:", sys.path[:6])
print("当前目录:", current_dir)

# ---------------------- 导入模块 ----------------------
try:
    # 配置
    from config.default import get_cfg
    cfg = get_cfg()
    print("✓ Config 导入成功")
    
    # 模型
    import model.vit_3d_detector as vit_module
    importlib.reload(vit_module)  # 重新加载避免缓存问题
    ViT3DDetector = vit_module.ViT3DDetector
    print("✓ ViT3DDetector 导入成功")
    
    # Trainer
    from training.trainer import Trainer
    print("✓ Trainer 导入成功")
    
    # InferenceEngine
    from inference.inference import InferenceEngine
    print("✓ InferenceEngine 导入成功")
    
except ImportError as e:
    print("❌ Import error:", e)
    # 打印文件信息方便排查
    print("当前目录文件:", os.listdir(current_dir))
    for folder in ['config', 'model', 'training', 'inference', 'utils']:
        print(f"{folder} 文件:", os.listdir(os.path.join(current_dir, folder)))
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='3D Object Detection with ViT')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                       help='Operation mode: train or inference')
    parser.add_argument('--data_root', type=str, 
                       help='Root directory of KITTI dataset (overrides config)')
    parser.add_argument('--image_path', type=str,
                       help='Path to input image for inference')
    parser.add_argument('--intrinsic', type=str,
                       help='Camera intrinsic parameters: "fx,fy,cx,cy"')
    parser.add_argument('--box2d', type=str,
                       help='2D bounding box: "min_x,min_y,max_x,max_y"')
    parser.add_argument('--model_path', type=str,
                       help='Path to model checkpoint (overrides config)')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training (overrides config)')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float,
                       help='Learning rate (overrides config)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for inference results')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize configuration
    cfg = get_cfg()
    
    # Override config with command line arguments if provided
    if args.data_root:
        cfg.DATA_ROOT = args.data_root
    if args.model_path:
        cfg.MODEL_PATH = args.model_path
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.lr:
        cfg.LR = args.lr
    
    print("Configuration:")
    print(f"  Data root: {cfg.DATA_ROOT}")
    print(f"  Model path: {cfg.MODEL_PATH}")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Epochs: {cfg.EPOCHS}")
    print(f"  Learning rate: {cfg.LR}")
    
    if args.mode == 'train':
        # Check if data directory exists
        if not os.path.exists(cfg.DATA_ROOT):
            print(f"Error: Data directory {cfg.DATA_ROOT} does not exist!")
            print("Please check the path and try again.")
            return
        
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(cfg.MODEL_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created model directory: {model_dir}")
        
        trainer = Trainer(cfg)
        
        # Resume training if requested and checkpoint exists
        if args.resume and os.path.exists(cfg.MODEL_PATH):
            print(f"Resuming training from checkpoint: {cfg.MODEL_PATH}")
            trainer.load_checkpoint(cfg.MODEL_PATH)
        elif args.resume and not os.path.exists(cfg.MODEL_PATH):
            print(f"Warning: Checkpoint {cfg.MODEL_PATH} not found, starting from scratch")
        
        trainer.train()
        
    elif args.mode == 'inference':
        if not all([args.image_path, args.intrinsic, args.box2d]):
            raise ValueError("For inference, image_path, intrinsic, and box2d must be provided")
        
        # Check if model exists
        if not os.path.exists(cfg.MODEL_PATH):
            print(f"Error: Model checkpoint {cfg.MODEL_PATH} does not exist!")
            print("Please train the model first or provide a valid model path.")
            return
        
        # Parse intrinsic and box2d parameters
        try:
            intrinsic = [float(x) for x in args.intrinsic.split(',')]
            if len(intrinsic) != 4:
                raise ValueError("Intrinsic parameters must be 4 values: fx,fy,cx,cy")
        except ValueError as e:
            raise ValueError(f"Invalid intrinsic parameters: {e}")
        
        try:
            box2d = [float(x) for x in args.box2d.split(',')]
            if len(box2d) != 4:
                raise ValueError("2D box must be 4 values: min_x,min_y,max_x,max_y")
        except ValueError as e:
            raise ValueError(f"Invalid 2D box parameters: {e}")
        
        print(f"Running inference on: {args.image_path}")
        print(f"Intrinsic parameters: {intrinsic}")
        print(f"2D bounding box: {box2d}")
        
        inference_engine = InferenceEngine(cfg, cfg.MODEL_PATH)
        result_image = inference_engine.run_inference(
            args.image_path, box2d, intrinsic
        )
        if isinstance(result_image, tuple):
    # 假设第一个元素是图像
          result_image = result_image[0]

      # 如果返回的是 numpy array，则转为 PIL Image
        if isinstance(result_image, np.ndarray):
          result_image = Image.fromarray(result_image)

        
        # Save result
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'result.jpg')
        result_image.save(output_path)
        print(f"Result saved to {output_path}")

if __name__ == '__main__':
    main()