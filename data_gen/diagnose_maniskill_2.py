"""
诊断 Part 3: ManiSkillScene API + TableSceneBuilder
    python diagnose_part3.py
"""
import inspect

# 1) ManiSkillScene 的方法
print("=" * 60)
print("[1] ManiSkillScene 可用方法 (含 'build', 'add', 'create', 'ground'):")
try:
    from mani_skill.envs.scene import ManiSkillScene
    methods = [x for x in dir(ManiSkillScene) if not x.startswith('__')]
    for m in methods:
        if any(k in m.lower() for k in ['build', 'add', 'create', 'ground', 'actor', 'table']):
            obj = getattr(ManiSkillScene, m)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"    {m}{sig}")
                except:
                    print(f"    {m} (callable)")
            else:
                print(f"    {m} = {type(obj)}")
except Exception as e:
    print(f"    Error: {e}")

# 2) TableSceneBuilder
print("\n[2] TableSceneBuilder:")
try:
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    print(f"    Found: {TableSceneBuilder}")
    sig = inspect.signature(TableSceneBuilder.__init__)
    print(f"    __init__{sig}")
    # 看 build 方法
    if hasattr(TableSceneBuilder, 'build'):
        src = inspect.getsource(TableSceneBuilder.build)
        lines = src.split('\n')[:40]
        print("    build() source (first 40 lines):")
        for line in lines:
            print(f"      {line}")
except ImportError:
    print("    Not found at mani_skill.utils.scene_builder.table")
    # 搜其他位置
    try:
        from mani_skill.utils.scene_builder import TableSceneBuilder
        print(f"    Found at: mani_skill.utils.scene_builder")
    except:
        pass
    try:
        from mani_skill.envs.utils.scene_builder import TableSceneBuilder
        print(f"    Found at: mani_skill.envs.utils.scene_builder")
    except:
        pass

# 3) create_actor_builder
print("\n[3] scene.create_actor_builder:")
try:
    from mani_skill.envs.scene import ManiSkillScene
    if hasattr(ManiSkillScene, 'create_actor_builder'):
        sig = inspect.signature(ManiSkillScene.create_actor_builder)
        print(f"    create_actor_builder{sig}")
    else:
        print("    NOT found")
        actor_methods = [x for x in dir(ManiSkillScene) if 'actor' in x.lower() or 'builder' in x.lower()]
        print(f"    Related methods: {actor_methods}")
except Exception as e:
    print(f"    Error: {e}")

# 4) 直接看 PickClutterYCB._load_scene 用了啥
print("\n[4] PickClutterYCB 的 _load_scene (完整):")
try:
    from mani_skill.envs.tasks.tabletop.pick_clutter_ycb import PickClutterYCBEnv
    src = inspect.getsource(PickClutterYCBEnv._load_scene)
    print(src)
except Exception as e:
    print(f"    Error: {e}")

# 5) 看 TableSceneBuilder 的 __init__ 参数
print("\n[5] TableSceneBuilder init 详细:")
try:
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    src = inspect.getsource(TableSceneBuilder.__init__)
    print(src)
except Exception as e:
    print(f"    Error: {e}")

print(f"\n{'='*60}")
print("  Done!")
print("=" * 60)