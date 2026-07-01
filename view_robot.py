import mujoco
import mujoco.viewer
import time

# 1. 加载机器人的 XML 模型文件 (把这里的名字换成你实际的文件名)
#model_path = "stardust_s1_with_dex_hand.xml"
model_path = "/Users/apple/vscodeProject/AVP/astribot_simulation/astribot_descriptions/mjcf/astribot_s1_mjcf/astribot_s1_with_hand.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("模型加载成功，正在启动可视化窗口...")

# 2. 启动 MuJoCo 原生的可视化交互窗口
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 3. 进入物理仿真循环
    while viewer.is_running():
        # 让物理引擎向前推演一步 (step)
        mujoco.mj_step(model, data)
        
        # 将最新的物理状态同步到显示画面
        viewer.sync()
        
        # 为了不让画面运行过快，稍微休眠一下 (MuJoCo 默认步长是 2 毫秒)
        time.sleep(0.0002)