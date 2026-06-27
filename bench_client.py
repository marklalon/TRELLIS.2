#!/usr/bin/env python3
"""TRELLIS.2 并发压力测试客户端

对 TRELLIS.2 推理服务器发起可配置并发数的请求，同时支持 WebSocket
（带进度流）和 HTTP（简单 multipart POST）两种协议，汇总统计信息。

用法:
    # 最基本用法: 4 个并发请求, 使用 example_image 目录下的图片
    python bench_client.py --concurrency 4

    # 指定服务器、协议、自定义图片
    python bench_client.py --server http://localhost:8086 --protocol ws ^
        --concurrency 8 --image-dir assets/example_image

    # 只跑 HTTP 协议, 单张图片
    python bench_client.py --protocol http --concurrency 1 ^
        --image assets/example_image/T.png

    # 限定图片数量 (随机选取 N 张)
    python bench_client.py --concurrency 20 --max-images 5

    # 输出 JSON 报告到文件
    python bench_client.py --concurrency 10 --report results.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# 统计容器
# --------------------------------------------------------------------------- #

@dataclass
class RequestResult:
    """单次请求的结果。"""
    ok: bool
    latency: float          # 端到端耗时 (秒)
    bytes_received: int = 0
    error: str = ""
    protocol: str = ""      # "ws" | "http"
    image: str = ""
    total_progress_stages: int = 0   # WS 收到的 progress 消息数


@dataclass
class Stats:
    """批量请求的统计摘要。"""
    total: int
    ok: int
    failed: int
    latencies: list[float]
    bytes_list: list[int]

    @property
    def latency_min(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def latency_max(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def latency_avg(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lats = sorted(self.latencies)
        idx = max(0, min(len(sorted_lats) - 1,
                         round(p / 100 * len(sorted_lats) - 0.5)))
        return sorted_lats[idx]

    @property
    def throughput(self) -> float:
        """吞吐量 = 成功数 / 最大延迟窗口"""
        total_time = self.latency_max if self.latencies else 0.0
        return self.total / total_time if total_time > 0 else 0.0

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_list)

    def report_lines(self) -> list[str]:
        return [
            "=" * 60,
            "  并发测试结果报告",
            "=" * 60,
            f"  总请求数:            {self.total}",
            f"  成功:                 {self.ok}",
            f"  失败:                 {self.failed}",
            f"  成功率:               {self.ok / max(self.total, 1) * 100:.1f}%",
            "",
            "  --- 延迟 (秒) ---",
            f"  最小值:               {self.latency_min:.2f}",
            f"  平均值:               {self.latency_avg:.2f}",
            f"  最大值:               {self.latency_max:.2f}",
            f"  P50 (中位数):         {self.percentile(50):.2f}",
            f"  P90:                  {self.percentile(90):.2f}",
            f"  P95:                  {self.percentile(95):.2f}",
            f"  P99:                  {self.percentile(99):.2f}",
            "",
            "  --- 吞吐 ---",
            f"  总耗时窗口:           {self.latency_max:.2f}s",
            f"  吞吐量:               {self.throughput:.2f} req/s",
            "",
            "  --- 输出 ---",
            f"  总接收字节数:         {self.total_bytes:,}",
            f"  平均输出大小:         {(self.total_bytes / max(self.ok, 1)):,.0f} bytes",
            "",
        ]


# --------------------------------------------------------------------------- #
# WebSocket 客户端
# --------------------------------------------------------------------------- #

async def _ws_request(
    server: str,
    image_path: str,
    params: dict,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """通过 WebSocket 发送一个请求，返回结果。"""
    try:
        import websockets
    except ImportError:
        return RequestResult(
            ok=False, protocol="ws", image=image_path, latency=0,
            error="missing 'websockets' package; pip install websockets",
        )

    async with semaphore:
        t0 = time.monotonic()
        ws_url = _ws_url(server)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        payload = {**params}

        try:
            async with websockets.connect(
                ws_url,
                max_size=64 * 1024 * 1024,
                open_timeout=timeout,
            ) as ws:
                await ws.send(json.dumps(payload))
                await ws.send(image_bytes)
                progress_count = 0
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        stage = msg.get("stage", "")
                        if stage == "queued":
                            continue
                        elif stage == "processing":
                            progress_count += 1
                            continue
                        elif stage == "done":
                            glb_bytes = await ws.recv()
                            elapsed = time.monotonic() - t0
                            return RequestResult(
                                ok=True, protocol="ws", image=image_path,
                                latency=elapsed, bytes_received=len(glb_bytes),
                                total_progress_stages=progress_count,
                            )
                        elif stage == "cancelled":
                            raise RuntimeError(msg.get("message", "cancelled"))
                        elif stage == "error":
                            raise RuntimeError(msg.get("message", "unknown"))
                except asyncio.CancelledError:
                    try:
                        await asyncio.shield(
                            ws.send(json.dumps({"type": "cancel"}))
                        )
                    except Exception:
                        pass
                    raise
                raise RuntimeError("connection closed before result")
        except asyncio.TimeoutError:
            return RequestResult(
                ok=False, protocol="ws", image=image_path,
                latency=time.monotonic() - t0, error="timeout",
            )
        except Exception as e:
            return RequestResult(
                ok=False, protocol="ws", image=image_path,
                latency=time.monotonic() - t0, error=str(e),
            )


# --------------------------------------------------------------------------- #
# HTTP 客户端
# --------------------------------------------------------------------------- #

async def _http_request(
    server: str,
    image_path: str,
    params: dict,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """通过 HTTP multipart POST 发送一个请求，返回结果。"""
    try:
        import aiohttp
    except ImportError:
        return RequestResult(
            ok=False, protocol="http", image=image_path,
            latency=0, error="missing 'aiohttp' package; pip install aiohttp",
        )

    async with semaphore:
        t0 = time.monotonic()
        url = server.rstrip("/") + "/generate"
        file_obj = open(image_path, "rb")
        try:
            data = aiohttp.FormData()
            data.add_field("image", file_obj,
                           filename=os.path.basename(image_path),
                           content_type="image/png")
            for k, v in params.items():
                data.add_field(k, str(v))

            timeout_ctx = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_ctx) as session:
                async with session.post(url, data=data) as resp:
                    body = await resp.read()
                    elapsed = time.monotonic() - t0
                    if resp.status == 200:
                        return RequestResult(
                            ok=True, protocol="http", image=image_path,
                            latency=elapsed, bytes_received=len(body),
                        )
                    else:
                        try:
                            detail = json.loads(body).get("error", body[:200])
                        except Exception:
                            detail = body[:200]
                        return RequestResult(
                            ok=False, protocol="http", image=image_path,
                            latency=elapsed,
                            error=f"HTTP {resp.status}: {detail}",
                        )
        except asyncio.TimeoutError:
            return RequestResult(
                ok=False, protocol="http", image=image_path,
                latency=time.monotonic() - t0, error="timeout",
            )
        except Exception as e:
            return RequestResult(
                ok=False, protocol="http", image=image_path,
                latency=time.monotonic() - t0, error=str(e),
            )
        finally:
            file_obj.close()


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #

def _ws_url(server: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    elif not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + "/ws/generate"


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _find_images(
    image_paths: list[str] | None,
    image_dir: str | None,
    max_images: int | None,
) -> list[str]:
    """收集测试用图片路径列表。"""
    images: list[str] = []

    if image_paths:
        for p in image_paths:
            if os.path.isfile(p):
                images.append(p)
            else:
                print(f"[warn] 图片不存在，跳过: {p}", file=sys.stderr)

    if image_dir and os.path.isdir(image_dir):
        for fname in sorted(os.listdir(image_dir)):
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTENSIONS:
                images.append(os.path.join(image_dir, fname))

    if not images:
        # fallback: 尝试默认路径
        default_dir = os.path.join(
            os.path.dirname(__file__), "assets", "example_image"
        )
        if os.path.isdir(default_dir):
            for fname in sorted(os.listdir(default_dir)):
                if os.path.splitext(fname)[1].lower() in _IMAGE_EXTENSIONS:
                    images.append(os.path.join(default_dir, fname))

    if not images:
        print("[error] 未找到任何测试图片!", file=sys.stderr)
        sys.exit(1)

    if max_images and len(images) > max_images:
        images = images[:max_images]

    return images


# --------------------------------------------------------------------------- #
# 主测试编排
# --------------------------------------------------------------------------- #

async def _run_bench(args) -> None:
    # --- 收集图片 ---
    images = _find_images(args.image, args.image_dir, args.max_images)

    # --- 生成参数 ---
    gen_params: dict = {
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        "simplify": args.simplify,
        "texture_sampling_steps": args.texture_sampling_steps,
        "shape_sampling_steps": args.shape_sampling_steps,
        "preprocess_image": True,
    }

    # --- 并发控制 ---
    # 总请求数 = 并发数, 每个并发一个请求
    sem = asyncio.Semaphore(args.concurrency)
    request_fn = _ws_request if args.protocol == "ws" else _http_request

    # --- 预热 (可选) ---
    if args.warmup > 0:
        print(f"[info] 预热: 发送 {args.warmup} 个请求 (串行)...")
        for i in range(args.warmup):
            img = images[i % len(images)]
            r = await request_fn(args.server, img, gen_params, args.timeout,
                                 asyncio.Semaphore(1))
            status = "OK" if r.ok else f"FAIL({r.error})"
            print(f"  warmup {i+1}/{args.warmup}: {status}  {r.latency:.2f}s")
        print("[info] 预热完成\n")

    # --- 启动测试 ---
    print(f"[info] 开始测试: protocol={args.protocol}  "
          f"concurrency={args.concurrency}  timeout={args.timeout}s  "
          f"images_available={len(images)}")
    print(f"[info] 参数: {json.dumps(gen_params, default=str)}")
    print()

    test_start = time.monotonic()

    # 构造任务列表 — 并发数 = 总请求数
    tasks = []
    for i in range(args.concurrency):
        img = images[i % len(images)]
        tasks.append(request_fn(args.server, img, gen_params, args.timeout, sem))

    # 并发执行并收集结果
    results: list[RequestResult] = []
    done_count = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done_count += 1

        status = "OK" if result.ok else "FAIL"
        if not result.ok:
            status += f"({result.error[:40]})"
        mb = result.bytes_received / 1024**2
        fname = os.path.basename(result.image)
        print(f"  [{done_count:>4}/{args.concurrency}] {status:50s}  "
              f"{result.latency:7.2f}s  {mb:.1f}MB  {fname}")

    test_elapsed = time.monotonic() - test_start

    # --- 统计 ---
    ok_results = [r for r in results if r.ok]
    failed_results = [r for r in results if not r.ok]
    stats = Stats(
        total=len(results),
        ok=len(ok_results),
        failed=len(failed_results),
        latencies=[r.latency for r in ok_results],
        bytes_list=[r.bytes_received for r in ok_results],
    )

    # 打印报告
    print()
    for line in stats.report_lines():
        print(line)
    print(f"  实际完成耗时:         {test_elapsed:.2f}s")
    print(f"  实际吞吐量:           {stats.total / test_elapsed:.2f} req/s")
    print()

    # 错误详情 (只打印前 10 条)
    if failed_results:
        n_show = min(10, len(failed_results))
        print(f"  失败详情 (前 {n_show} 条):")
        for i, r in enumerate(failed_results[:n_show]):
            print(f"    {i+1}. [{r.protocol}] {os.path.basename(r.image)}: "
                  f"{r.error[:100]}")
        if len(failed_results) > 10:
            print(f"    ... 还有 {len(failed_results) - 10} 条")
        print()

    # 输出 JSON 报告
    if args.report:
        report = {
            "meta": {
                "server": args.server,
                "protocol": args.protocol,
                "concurrency": args.concurrency,
                "timeout": args.timeout,
                "pipeline_type": args.pipeline_type,
                "warmup": args.warmup,
            },
            "summary": {
                "total": stats.total,
                "ok": stats.ok,
                "failed": stats.failed,
                "latency_min": round(stats.latency_min, 3),
                "latency_avg": round(stats.latency_avg, 3),
                "latency_max": round(stats.latency_max, 3),
                "latency_p50": round(stats.percentile(50), 3),
                "latency_p90": round(stats.percentile(90), 3),
                "latency_p95": round(stats.percentile(95), 3),
                "latency_p99": round(stats.percentile(99), 3),
                "throughput_req_per_sec": round(stats.throughput, 2),
                "wall_clock_sec": round(test_elapsed, 2),
                "total_bytes": stats.total_bytes,
                "avg_bytes_per_ok": round(stats.total_bytes / max(stats.ok, 1)),
            },
            "errors": [
                {"image": os.path.basename(r.image), "error": r.error[:200]}
                for r in failed_results
            ],
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[info] 报告已保存到: {args.report}")
        print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 并发压力测试客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python bench_client.py --concurrency 4
  python bench_client.py --protocol http --concurrency 2
  python bench_client.py --image img1.png img2.png --concurrency 8
  python bench_client.py --concurrency 10 --report result.json
        """,
    )

    # 连接
    parser.add_argument("--server", default="http://localhost:8086",
                        help="服务器地址 (default: http://localhost:8086)")
    parser.add_argument("--protocol", choices=["ws", "http"], default="ws",
                        help="通信协议 (default: ws)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="单请求超时秒数 (default: 600)")

    # 负载
    parser.add_argument("--concurrency", "-c", type=int, default=4,
                        help="并发数 / 总请求数 (default: 4)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="预热请求数 (串行, default: 0)")

    # 图片
    parser.add_argument("--image", "-i", nargs="*",
                        help="图片路径 (可指定多张)")
    parser.add_argument("--image-dir", "-d",
                        default=os.path.join(
                            os.path.dirname(__file__), "assets", "example_image",
                        ),
                        help="图片目录 (default: assets/example_image)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="最多使用 N 张不同图片 (default: 全部)")

    # 生成参数
    parser.add_argument("--pipeline-type", default="512",
                        help="pipeline 类型 (default: 512)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument("--texture-sampling-steps", type=int, default=12)
    parser.add_argument("--shape-sampling-steps", type=int, default=12)
    parser.add_argument("--simplify", type=int, default=1000000)

    # 输出
    parser.add_argument("--report", default=None,
                        help="输出 JSON 报告路径 (default: 不输出)")

    args = parser.parse_args()

    try:
        asyncio.run(_run_bench(args))
    except KeyboardInterrupt:
        print("\n[bench] 测试被用户中断", file=sys.stderr)
        raise SystemExit(130)
    except Exception as e:
        print(f"[bench] 错误: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
