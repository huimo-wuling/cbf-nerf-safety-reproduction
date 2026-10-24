# -*- coding: utf-8 -*-
"""run_and_tee.py —— 运行上游实验,同时把原始输出写入日志、把进度打印到终端

为什么需要
----------
上游脚本把指标全部 `print` 到标准输出,但实验要跑十几分钟。若直接把输出重定向进
日志文件,终端在整个过程中没有任何反馈;若只输出到终端,又拿不到日志给
`tools/plot_from_log.py` 解析。

本脚本把两者合并:子进程的每一行原样追加到日志(供解析),同时在终端只打印一行
紧凑的进度(每帧一条),避免刷屏。

用法(由 run_upstream.bat 调用)::

    python tools/run_and_tee.py <日志文件> <上游脚本> [脚本参数...]

子进程的工作目录自动设为上游脚本所在目录(上游用相对路径读 configs/、device.txt)。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

NUM = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
RE_MIN_SINGLE = re.compile(r'min_depth\s*=\s*(?:tensor\()?(%s)' % NUM)
RE_MIN_DOUBLE = re.compile(r'min depth\s*=\s*(?:tensor\()?(%s)' % NUM)
RE_STEP = re.compile(r'^\s*step:\s*(\d+)\s*$')
RE_FRAME = re.compile(r'^\s*(\d+)\s*$')
RE_IV = re.compile(r'intervention\s*=\s*(%s)' % NUM)
RE_V = re.compile(r'new v\s*=\s*(%s)' % NUM)
RE_T = re.compile(r'time cost\s*=\s*(%s)' % NUM)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print('usage: run_and_tee.py <logfile> <script> [script args...]', file=sys.stderr)
        return 2
    log_path, script, *script_args = argv
    script = Path(script).resolve()
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:                                   # 终端中文不乱码
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

    env = dict(os.environ, PYTHONUNBUFFERED='1')
    is_double = 'double' in script.name.lower()
    total = 100 if is_double else 150

    print('== launch: %s' % script.name)
    print('== log   : %s' % log_path)
    print('== 每帧一行进度;完整输出写入上面的日志文件')
    print('-' * 66)

    t0 = time.time()
    n_frame = 0
    pend_min = pend_v = pend_t = None
    last_iv = 0.0
    min_seen = None
    h_max = None
    with open(log_path, 'w', encoding='utf-8', newline='') as fout:
        proc = subprocess.Popen([sys.executable, '-u', str(script), *script_args],
                                cwd=str(script.parent), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in proc.stdout:
            fout.write(line)
            fout.flush()
            s = line.rstrip('\n')

            m = RE_IV.search(s)
            if m:
                last_iv = float(m.group(1))
                continue
            if is_double:
                m = RE_STEP.search(s)
                if m:
                    n_frame = int(m.group(1)) + 1
                    continue
                m = RE_MIN_DOUBLE.search(s)
                if m:
                    pend_min = float(m.group(1))
                    continue
                m = RE_V.search(s)
                if m:
                    pend_v = float(m.group(1))
                    continue
                m = RE_T.search(s)
                if m:
                    pend_t = float(m.group(1))
                    if pend_min is not None:
                        h = 0.1 - pend_min + (pend_v or 0.0)
                        min_seen = pend_min if min_seen is None else min(min_seen, pend_min)
                        print('[%3d/%d]  min_depth=%.4f  h=%+.4f  v=%.3f  iv=%.2f  t=%.2fs'
                              % (n_frame, total, pend_min, h, pend_v or 0.0, last_iv, pend_t or 0.0),
                              flush=True)
                    pend_min = pend_v = pend_t = None
                continue
            m = RE_FRAME.search(s)
            if m:
                n_frame = int(m.group(1))
                continue
            m = RE_MIN_SINGLE.search(s)
            if m:
                md = float(m.group(1))
                min_seen = md if min_seen is None else min(min_seen, md)
                tag = 'init ' if n_frame == 0 else '%3d/%d' % (n_frame, total)
                print('[%s]  min_depth=%.4f  iv=%.2f' % (tag, md, last_iv), flush=True)
                continue
        rc = proc.wait()

    dt = time.time() - t0
    print('-' * 66)
    print('== finished: %s  用时 %.1f 分钟  退出码 %d'
          % (script.name, dt / 60, rc))
    if min_seen is not None:
        print('== 最小最近距离: %.4f m（d_c = 0.1 m）' % min_seen)
    return rc


if __name__ == '__main__':
    sys.exit(main())
