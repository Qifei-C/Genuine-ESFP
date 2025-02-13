from uarm.wrapper import SwiftAPI
import time


def check_arm_limits(api, x_range, y_range, z_range, step=10):
    """
    检测机械臂的运动范围和限位
    :param api: SwiftAPI 对象，用于控制机械臂
    :param x_range: X 轴移动范围 [min_x, max_x]
    :param y_range: Y 轴移动范围 [min_y, max_y]
    :param z_range: Z 轴移动范围 [min_z, max_z]
    :param step: 每次检测移动的步长（单位 mm）
    :return: 测试的结果
    """
    results = {"success": True, "errors": []}

    # 获取当前机械臂状态
    if not api.connected:
        print("机械臂未连接，正在尝试连接中...")
        api.connect()

    print("开始检测机械臂范围...")
    time.sleep(1)

    # 遍历测试的目标范围 (X, Y, Z)
    for x in range(x_range[0], x_range[1] + step, step):
        for y in range(y_range[0], y_range[1] + step, step):
            for z in range(z_range[0], z_range[1] + step, step):
                print(f"移动到位置: X={x}, Y={y}, Z={z}")
                try:
                    # 设置机械臂移动到指定位置
                    response = api.set_position(x=x, y=y, z=z, wait=True, timeout=5)

                    # 检查是否返回异常
                    if response == 'TIMEOUT' or response is None:
                        raise ValueError(f"机械臂未能到达位置: X={x}, Y={y}, Z={z}")

                    # 获取实际到达的位置
                    current_position = api.get_position()
                    print(f"当前到达的位置: {current_position}")

                except Exception as e:
                    # 记录错误
                    error_message = f"限位检测失败 -> X={x}, Y={y}, Z={z}: {e}"
                    print(error_message)
                    results["errors"].append(error_message)

    # 检测完成
    if results["errors"]:
        results["success"] = False
        print("检测过程中存在错误！")
    else:
        print("机械臂范围和限位检测成功！无错误。")

    return results


if __name__ == "__main__":
    # 定义机械臂运动范围 (可以根据具体的机械臂规格调整)
    X_RANGE = [100, 300]  # X方向的移动范围 (mm)
    Y_RANGE = [-150, 150]  # Y方向的移动范围 (mm)
    Z_RANGE = [0, 250]  # Z方向的移动范围 (mm)

    # 初始化机械臂对象
    arm = SwiftAPI()

    # 调用检测函数
    result = check_arm_limits(arm, x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE)

    # 输出检测结果
    if result["success"]:
        print("机械臂范围检测完成，所有范围检测通过！")
    else:
        print("机械臂范围检测失败，以下是错误详情：")
        for error in result["errors"]:
            print(error)

    # 断开连接
    arm.disconnect()
    print("机械臂已断开连接。")