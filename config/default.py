class Config:
    def __init__(self):
        # Data configuration
        self.DATA_ROOT = '/Users/chenyufeng/desktop/kitti_tiny_3D'
        self.IMAGE_SIZE = (224, 224)
        
        # Model configuration
        self.BACKBONE = 'vit_base_patch16_224'
        self.HIDDEN_DIM = 256
        self.NUM_QUERIES = 100
        self.MODEL_PATH = '/Users/chenyufeng/desktop/3d_object_detection/checkpoints/model.pth'
        
        # Training configuration
        self.BATCH_SIZE = 8
        self.EPOCHS = 50
        self.LR = 1e-4
        self.WEIGHT_DECAY = 1e-4
        
        # Loss weights
        self.LOSS_WEIGHTS = {
            'center': 1.0,
            'size': 0.5,
            'angle': 0.2
        }

def get_cfg():
    return Config()