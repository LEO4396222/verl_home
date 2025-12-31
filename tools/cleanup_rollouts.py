#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

'''
示例：python /data/huaiwenzhang/projects/verl/tools/cleanup_rollouts.py \
    /data/huaiwenzhang/projects/verl/training_rollout_metrics/dapo_advclip_baseline_no_overlong_penalty_follow_cov \
    --keep-every 10 \
    --apply
'''
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="保留每隔N个step的Rollout文件，删除其余文件（默认dry-run）"
    )
    parser.add_argument(
        "dir",
        help="Rollout目录路径",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=10,
        help="保留间隔（默认10）",
    )
    parser.add_argument(
        "--pattern",
        default=r"^(\d+)\.jsonl$",
        help="文件名正则，需包含step数字（默认: ^(\\d+)\\.jsonl$）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出全部待删除文件",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行删除（默认仅dry-run）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"目录不存在: {dir_path}")
        return 1
    if args.keep_every <= 0:
        print("keep-every 必须是正整数")
        return 1

    pattern = re.compile(args.pattern)

    matched = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        matched.append((step, p))

    matched.sort(key=lambda x: x[0])

    keep_files = []
    delete_files = []
    for step, p in matched:
        if step % args.keep_every == 0:
            keep_files.append(p)
        else:
            delete_files.append(p)

    print(f"扫描目录: {dir_path}")
    print(f"匹配到文件数: {len(matched)}")
    print(f"保留数: {len(keep_files)}")
    print(f"删除数: {len(delete_files)}")

    if args.list:
        print("待删除文件列表:")
        for p in delete_files:
            print(p.name)
    else:
        preview = delete_files[:20]
        if preview:
            print("待删除预览(前20个):")
            for p in preview:
                print(p.name)
            rest = len(delete_files) - len(preview)
            if rest > 0:
                print(f"...还有 {rest} 个未显示，可用 --list 查看")

    if args.apply:
        for p in delete_files:
            p.unlink()
        print("已删除完毕")
    else:
        print("dry-run 模式：未删除文件，如需删除请加 --apply")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
