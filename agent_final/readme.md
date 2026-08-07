# 1. 启动云端（选择协助模式）
CUDA_VISIBLE_DEVICES=0 python cloud_agent_v2.py --mode lightweight --port 5000
# 或
python cloud_agent_v2.py --mode model_reprovision --port 5000
# 或
python cloud_agent_v2.py --mode auto --port 5000

# 2. 启动边缘端
# 离线运行（自动检测权重：有则测试，无则训练）
python edge_agent_v2.py --config config.yaml

# 连接云端
python edge_agent_v2.py --config config.yaml --cloud --cloud-server http://localhost:5000