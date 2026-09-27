"""
碳金融市场ABM仿真 - 快速启动脚本
提供简单的交互式入口
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行主程序
from carbon_market_abm.main import main

if __name__ == "__main__":
    main()
