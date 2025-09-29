import json
from collections import defaultdict
import argparse

def analyze_json(file_path):
    # 用于存储每种op_name的统计信息
    op_stats = defaultdict(lambda: {"count": 0, "total_dur": 0, "provider": set()})
    total_duration = 0

    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 遍历JSON数据
    for item in data:
        if isinstance(item, dict) and 'name' in item and 'args' in item and 'op_name' in item['args']:
            op_name = item['args']['op_name']
            dur = item['dur']
            provider = item['args'].get('provider', 'Unknown')

            # 更新统计信息
            op_stats[op_name]['count'] += 1
            op_stats[op_name]['total_dur'] += dur
            op_stats[op_name]['provider'].add(provider)
            total_duration += dur

    # 按总耗时降序排序
    sorted_stats = sorted(op_stats.items(), key=lambda x: x[1]['total_dur'], reverse=True)

    # 计算每列的最大宽度
    max_op_name = max(len(op_name) for op_name, _ in sorted_stats)
    max_count = max(len(str(stats['count'])) for _, stats in sorted_stats)
    max_dur = max(len(f"{stats['total_dur']:,}") for _, stats in sorted_stats)
    max_provider = max(len(', '.join(stats['provider'])) for _, stats in sorted_stats)

    # 输出结果
    print(f"{'Op Name':<{max_op_name}} | {'Count':>{max_count}} | {'Total Duration (us)':>{max_dur}} | {'Provider':<{max_provider}}")
    print("-" * (max_op_name + max_count + max_dur + max_provider + 9))
    for op_name, stats in sorted_stats:
        print(f"{op_name:<{max_op_name}} | {stats['count']:>{max_count}} | {stats['total_dur']:>{max_dur},} | {', '.join(stats['provider']):<{max_provider}}")
     # 输出所有算子的总耗时
    print("\n" + "=" * (max_op_name + max_count + max_dur + max_provider + 9))
    print(f"Total duration of all operators: {total_duration:,} us")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze JSON trace file for op statistics.")
    parser.add_argument("file_path", help="Path to the JSON trace file")
    args = parser.parse_args()

    analyze_json(args.file_path)

