class Config:
    def __init__(self):
        # Data configuration
        self.DATA_ROOT = '/content/3d_detection/kitti_tiny_3D'
        self.IMAGE_SIZE = (448, 448)
        
        # Model configuration
        self.BACKBONE = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
        self.HIDDEN_DIM = 384
        self.NUM_QUERIES = 50
        self.MODEL_PATH = '/content/3d_detection/checkpoints/eva02_model.pth'
        
        # Training configuration
        self.BATCH_SIZE = 4  # 减小batch size
        self.EPOCHS = 100
        self.LR = 1e-4  # 增大学习率
        self.WEIGHT_DECAY = 1e-4
        
        # 🔥 调整损失权重（降低所有权重）
        self.LOSS_WEIGHTS = {
            'center': 0.5,   # 从2.0降到0.5
            'size': 0.3,     # 从1.0降到0.3
            'angle': 0.3,    # 从1.5降到0.3
            'depth': 1.0,    # 从3.0降到1.0
            'conf': 0.1      # 从0.5降到0.1
        }
        
        # 兼容性
        self.DATA = type('DATA', (), {
            'ROOT': self.DATA_ROOT,
            'IMAGE_SIZE': self.IMAGE_SIZE,
            'NUM_WORKERS': 0
        })()
        
        self.MODEL = type('MODEL', (), {
            'BACKBONE': self.BACKBONE,
            'HIDDEN_DIM': self.HIDDEN_DIM,
            'NUM_QUERIES': self.NUM_QUERIES,
            'PATH': self.MODEL_PATH
        })()
        
        self.TRAIN = type('TRAIN', (), {
            'BATCH_SIZE': self.BATCH_SIZE,
            'EPOCHS': self.EPOCHS,
            'LR': self.LR,
            'WEIGHT_DECAY': self.WEIGHT_DECAY
        })()
        
        self.OUTPUT_DIR = '/content/3d_detection/outputs'

def get_cfg():
    return Config()