# 仿真 100 Hz sample clock。
# action = 最新且 stamp <= t_k 的 active joint command
# state  = 最近 measured state
# image  = 最近 robot-view frame
# 保存每种模态原始时间戳与 sample_time。
