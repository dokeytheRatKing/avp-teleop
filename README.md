# 1. clone 我的代码
git clone https://github.com/YOUR_GITHUB_USERNAME/avp-teleop.git
cd avp-teleop

# 2. clone 机器人模型（运行时必须）
git clone https://github.com/Astribot-Dev/astribot_simulation.git

# 3. （可选）clone 参考实现，用于理解合并求解器
git clone https://github.com/Improbable-AI/VisionProTeleop.git

# 4. 建conda 环境
conda env create -f environment.yml
conda activate AVP

# 5. 自检（不需要 Vision Pro硬件）
python -m avp_teleop_upper_body.selfcheck
# 应看到 9/9 checks passed.
