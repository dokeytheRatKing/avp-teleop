import time
import numpy as np
from avp_stream import VisionProStreamer

def main():
    # 步骤 1：建立与 Apple Vision Pro 的网络传输握手实例
    # 需将此字符串替换为佩戴 AVP 后启动 Tracking Streamer App 时屏幕显示的 IP 地址或六位房间码
    #avp_connection_id = "192.168.1.100" 
    avp_connection_id = "10.200.177.142" 
    
    print(f"正在启动遥操作控制流，尝试连接至空间计算节点: {avp_connection_id}...")
    # 初始化 Streamer 类，该过程将自动拉起底层的 gRPC 侦听线程池与可能存在的 WebRTC 协商逻辑
    streamer = VisionProStreamer(ip=avp_connection_id)
    
    print("底层网络链路握手成功，等待 ARKit 手部追踪数据流高频注入...")
    
    # 步骤 2：定义核心的骨骼降维拓扑映射数组
    # 从 ARKit 原生的 25 关节索引中，精准剥离 4 根手指深层的掌骨（Metacarpal）节点
    # 剔除的索引号为：5（食指掌骨）, 10（中指掌骨）, 15（无名指掌骨）, 20（小指掌骨）
    target_21_indices = [
        0,              # 核心基座：腕关节
        1, 2, 3, 4,     # 驱动链 1：拇指全链路
        6, 7, 8, 9,     # 驱动链 2：食指去除掌骨后的有效操作关节
        11, 12, 13, 14, # 驱动链 3：中指去除掌骨后的有效操作关节
        16, 17, 18, 19, # 驱动链 4：无名指去除掌骨后的有效操作关节
        21, 22, 23, 24  # 驱动链 5：小指去除掌骨后的有效操作关节
    ]
    
    try:
        # 进入无限循环控制环路，此范式为具身智能控制周期的标准架构
        while True:
            # 步骤 3：非阻塞式提取缓冲队列中最新一帧的时序数据字典
            current_frame_data = streamer.latest
            
            # 鲁棒性保护：若由于网络丢包、操作者头部剧烈晃动导致追踪丢失，
            # 系统应温和地自旋等待，坚决避免抛出空指针异常导致系统崩溃
            if current_frame_data is None:
                time.sleep(0.01)
                continue
                
            # 从主数据字典中，基于键值提取左右手原生的三维张量，其原始形状必为 (25, 4, 4)
            raw_left_tensor = current_frame_data.get('left_fingers')
            raw_right_tensor = current_frame_data.get('right_fingers')
            
            # 完整性校验：必须确保 AVP 的广角摄像头阵列完整捕获了双手的形体轮廓
            if raw_left_tensor is not None and raw_right_tensor is not None:
                
                # 步骤 4：基于预定义的拓扑数组，执行张量的高级切片映射
                # 此步操作将原生数据从 (25, 4, 4) 空间无损降维至工业标准的 (21, 4, 4)
                filtered_left_poses = raw_left_tensor[target_21_indices, :, :]
                filtered_right_poses = raw_right_tensor[target_21_indices, :, :]
                
                # 步骤 5：空间平移向量抽取
                # 从 SE(3) 齐次变换矩阵中，剥离描述旋转的 3x3 子矩阵，
                # 仅保留描述三维空间绝对坐标（X, Y, Z）的第 4 列向量
                # 最终获得的变量形状为极其纯净的 (21, 3) 二维数组
                left_21_coordinates = filtered_left_poses[:, 0:3, 3]
                right_21_coordinates = filtered_right_poses[:, 0:3, 3]
                
                # 步骤 6：终端界面渲染控制
                # 利用 ANSI 转义序列 '\033:")
                '''
                for index, coord in enumerate(left_21_coordinates):
                    # 格式化输出浮点数至小数点后 4 位，保障毫米级观测精度
                    #print(f"  节点 {index:02d}: [{coord:+.4f}, {coord:+.4f}, {coord:+.4f}]")
                    
                    
                print("\n:")
                for index, coord in enumerate(right_21_coordinates):
                    #print(f"  节点 {index:02d}: [{coord:+.4f}, {coord:+.4f}, {coord:+.4f}]")
                    print(f"  节点 {index:02d}: [{coord:+.4f}, {coord[1]:+.4f}, {coord[2]:+.4f}]")
                '''

                print(left_21_coordinates)
                print("\n:")
                print(right_21_coordinates)

                print("\n>>> 系统运行状态: 追踪正常 | 数据流稳定流转中...")
            else:
                # 异常反馈，指导操作者矫正物理姿态
                print("\r>>> 追踪异常：未完整检测到双臂，请将双手移至 AVP 正前方视场角内...   ", end="")
                
            # 步骤 7：控制环路频率限流器 (Throttle)
            # 强制令 Mac M5 的线程休眠约 16.6 毫秒，使其数据摄取频率维持在稳定且充裕的 60Hz
            time.sleep(1/60)
            
    except KeyboardInterrupt:
        # 捕获用户通过 Ctrl+C 触发的退出中断，优雅关闭数据流隧道
        print("\n\n>>> 接收到手动中断信号，四十二位姿流式传输控制系统已安全退出。")

if __name__ == "__main__":
    main()