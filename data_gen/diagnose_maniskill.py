"""
OccluBench 诊断: 确认 ManiSkill3 的 BaseEnv API
在你的 occluvla-sim 环境中运行:
    python diagnose_maniskill.py
"""
import inspect

print("=" * 60)
print("  ManiSkill3 API 诊断")
print("=" * 60)

# 1) 版本
try:
    import mani_skill
    print(f"\n[1] mani_skill version: {mani_skill.__version__}")
except Exception as e:
    print(f"\n[1] mani_skill import error: {e}")

# 2) BaseEnv._default_sim_config 的返回类型
try:
    from mani_skill.envs.sapien_env import BaseEnv
    sig = inspect.signature(BaseEnv._default_sim_config.fget)
    print(f"\n[2] BaseEnv._default_sim_config return annotation: {sig.return_annotation}")

    # 看看有没有 SimConfig
    try:
        from mani_skill.envs.sapien_env import SimConfig
        print(f"    SimConfig found in sapien_env: {SimConfig}")
        print(f"    SimConfig fields: {list(SimConfig.__fields__.keys()) if hasattr(SimConfig, '__fields__') else 'N/A'}")
    except ImportError:
        pass

    try:
        from mani_skill.utils.structs.types import SimConfig
        print(f"    SimConfig found in structs.types: {SimConfig}")
    except ImportError:
        pass

    try:
        from mani_skill.envs.scene import SimConfig
        print(f"    SimConfig found in envs.scene: {SimConfig}")
    except ImportError:
        pass

    # 直接看 BaseEnv 源码中 _default_sim_config 怎么写的
    src = inspect.getsource(BaseEnv._default_sim_config.fget)
    print(f"\n    BaseEnv._default_sim_config source:\n{src}")
except Exception as e:
    print(f"\n[2] Error: {e}")

# 3) 看一个能跑的环境 (PickClutterYCB) 怎么定义的
print("\n[3] PickClutterYCB-v1 的 _default_sim_config:")
try:
    from mani_skill.envs.tasks.tabletop.pick_clutter_ycb import PickClutterYCBEnv
    if hasattr(PickClutterYCBEnv, '_default_sim_config'):
        src = inspect.getsource(PickClutterYCBEnv._default_sim_config.fget)
        print(f"    Source:\n{src}")
    else:
        print("    Not overridden (uses BaseEnv default)")
except Exception as e:
    print(f"    Error: {e}")

# 4) SUPPORTED_ROBOTS 和 robot_uids 机制
print("\n[4] Robot UID 机制:")
try:
    src_init = inspect.getsource(BaseEnv.__init__)
    # 找 robot_uids 相关行
    for line in src_init.split('\n'):
        if 'robot_uid' in line.lower() or 'SUPPORTED_ROBOTS' in line:
            print(f"    {line.strip()}")
except Exception as e:
    print(f"    Error: {e}")

# 看 PickClutterYCB 怎么设 SUPPORTED_ROBOTS
try:
    print(f"    PickClutterYCB.SUPPORTED_ROBOTS = {PickClutterYCBEnv.SUPPORTED_ROBOTS}")
    if hasattr(PickClutterYCBEnv, '_default_robot_uids'):
        print(f"    PickClutterYCB._default_robot_uids = {PickClutterYCBEnv._default_robot_uids}")
    if hasattr(PickClutterYCBEnv, 'SUPPORTED_ROBOTS'):
        print(f"    type: {type(PickClutterYCBEnv.SUPPORTED_ROBOTS)}")
except Exception as e:
    print(f"    Error: {e}")

# 5) gym.make 的 robot_uids 参数
print("\n[5] gym.make 参数测试:")
try:
    import gymnasium as gym
    env = gym.make(
        "PickClutterYCB-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        max_episode_steps=5,
    )
    uw = env.unwrapped
    print(f"    robot type: {type(uw.agent)}")
    print(f"    robot name: {uw.agent.robot.name if hasattr(uw.agent, 'robot') else 'N/A'}")
    print(f"    robot_uids used: {uw.robot_uids if hasattr(uw, 'robot_uids') else 'N/A'}")
    env.close()
except Exception as e:
    print(f"    Error: {e}")

# 6) build_actor_ycb 确认
print("\n[6] build_actor_ycb:")
try:
    from mani_skill.utils.building import actors as actor_utils
    print(f"    build_actor_ycb exists: {hasattr(actor_utils, 'build_actor_ycb')}")
    if hasattr(actor_utils, 'build_actor_ycb'):
        sig = inspect.signature(actor_utils.build_actor_ycb)
        print(f"    Signature: {sig}")
    else:
        print(f"    Available: {[x for x in dir(actor_utils) if 'ycb' in x.lower() or 'build' in x.lower()]}")
except Exception as e:
    print(f"    Error: {e}")

# 7) CameraConfig mount 参数
print("\n[7] CameraConfig:")
try:
    from mani_skill.sensors.camera import CameraConfig
    sig = inspect.signature(CameraConfig)
    print(f"    CameraConfig params: {list(sig.parameters.keys())}")
    if 'mount' in sig.parameters:
        print(f"    mount param: {sig.parameters['mount']}")
    if 'articulation_link' in sig.parameters:
        print(f"    articulation_link param: {sig.parameters['articulation_link']}")
except Exception as e:
    print(f"    Error: {e}")

print(f"\n{'='*60}")
print("  Done! 把上面的输出贴给我，我来修 occlubench_env.py")
print("=" * 60)