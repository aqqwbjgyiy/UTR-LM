import os

class Config:
    def __init__(self):
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 数据相关路径
        self.data_dir = os.path.join(self.project_root, 'Data', 'MRL_Random50Nuc_SynthesisLibrary_Sample')
        
        # 输出相关路径
        self.output_dir = os.path.join(self.project_root, 'Sample')
        self.model_dir = os.path.join(self.output_dir, 'saved_models')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs = [self.data_dir, self.model_dir, self.figure_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

config = Config()