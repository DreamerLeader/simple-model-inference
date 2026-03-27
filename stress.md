⏺ stress_cluster_benchmark.py 使用说明

  这是一个 Mooncake 分布式存储的零拷贝批量读写压测工具，采用 prefill/decode（生产者/消费者）模式。

  ---
  前置条件

  运行前需要确保以下服务已启动：
  1. Master 服务 — 默认监听 localhost:50051
  2. Metadata 服务 — 默认监听 http://127.0.0.1:8080/metadata
  3. RDMA 设备可用（或使用 TCP 协议）

  ---
  参数说明

  ┌───────────────────────┬────────────────────────────────┬───────────────────────────────┐
  │         参数          │             默认值             │             说明              │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --role                │ 必填                           │ prefill(写入) 或 decode(读取) │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --protocol            │ rdma                           │ 通信协议，rdma 或 tcp         │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --device-name         │ erdma_0                        │ RDMA 网卡设备名               │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --local-hostname      │ localhost                      │ 本机主机名/IP                 │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --metadata-server     │ http://127.0.0.1:8080/metadata │ 元数据服务地址                │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --master-server       │ localhost:50051                │ Master 服务地址               │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --global-segment-size │ 10000                          │ 全局段大小 (MB)               │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --local-buffer-size   │ 512                            │ 本地缓冲区大小 (MB)           │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --max-requests        │ 1200                           │ 总请求数                      │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --value-length        │ 4194304 (4MB)                  │ 每个 value 的大小 (字节)      │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --batch-size          │ 1                              │ 每批操作的 key 数量           │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --wait-time           │ 20                             │ 操作完成后等待时间 (秒)       │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --num-workers         │ 1                              │ 并发 worker 线程数            │
  ├───────────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ --detailed-stats      │ false                          │ 打印每个 worker 的详细统计    │
  └───────────────────────┴────────────────────────────────┴───────────────────────────────┘

  ---
  使用方式

  这个脚本需要在 两个终端 分别运行 prefill 和 decode，模拟写入端和读取端。

  1. 基础用法 — 单线程写入 + 读取

  终端 1（先启动写入端）：
  python stress_cluster_benchmark.py \
      --role prefill \
      --protocol rdma \
      --device-name erdma_0 \
      --local-hostname 192.168.1.10 \
      --metadata-server http://127.0.0.1:8080/metadata \
      --master-server 127.0.0.1:50051 \
      --max-requests 1200 \
      --value-length 4194304 \
      --batch-size 1

  终端 2（写入完成前启动读取端）：
  python stress_cluster_benchmark.py \
      --role decode \
      --protocol rdma \
      --device-name erdma_0 \
      --local-hostname 192.168.1.10 \
      --metadata-server http://127.0.0.1:8080/metadata \
      --master-server 127.0.0.1:50051 \
      --max-requests 1200 \
      --value-length 4194304 \
      --batch-size 1

  2. 多线程并发写入

  python stress_cluster_benchmark.py \
      --role prefill \
      --protocol rdma \
      --device-name erdma_0 \
      --local-hostname 192.168.1.10 \
      --metadata-server http://127.0.0.1:8080/metadata \
      --master-server 127.0.0.1:50051 \
      --max-requests 2400 \
      --value-length 4194304 \
      --batch-size 4 \
      --num-workers 4 \
      --detailed-stats

  --num-workers 4 启动 4 个线程，2400 个请求会均分给每个 worker（各 600 个）。--detailed-stats 会输出每个 worker
  的独立统计。

  3. 使用 TCP 协议（无 RDMA 环境）

  python stress_cluster_benchmark.py \
      --role prefill \
      --protocol tcp \
      --device-name "" \
      --local-hostname 127.0.0.1 \
      --metadata-server http://127.0.0.1:8080/metadata \
      --master-server 127.0.0.1:50051 \
      --max-requests 500 \
      --value-length 1048576 \
      --batch-size 2

  4. 大批量高吞吐测试

  python stress_cluster_benchmark.py \
      --role prefill \
      --protocol rdma \
      --device-name erdma_0 \
      --local-hostname 192.168.1.10 \
      --metadata-server http://127.0.0.1:8080/metadata \
      --master-server 127.0.0.1:50051 \
      --max-requests 10000 \
      --value-length 1048576 \
      --batch-size 16 \
      --num-workers 8 \
      --wait-time 5 \
      --detailed-stats

  ---
  工作原理

  prefill (写入端)                    decode (读取端)
     │                                    │
     ├─ setup() 连接 master               ├─ setup() 连接 master
     ├─ register_buffer() 注册内存        ├─ register_buffer() 注册内存
     ├─ batch_put_from() 批量写入 ──────► ├─ batch_get_into() 批量读取
     │   key0, key1, ... keyN             │   key0, key1, ... keyN
     └─ 输出性能统计                      └─ 输出性能统计

  - prefill: 向每个 key 写入填充数据（batch_put_from），数据通过 RDMA/TCP 零拷贝写入全局段
  - decode: 从相同的 key 读取数据（batch_get_into），通过 RDMA/TCP 零拷贝读取
  - 两端需要使用 相同的 --max-requests、--value-length、--batch-size 保证 key 名一致

  输出指标

  运行结束后会输出：
  - 延迟: mean / min / max / P90 / P99 / P999
  - 吞吐: ops/s 和 MB/s（分 operation time 和 wall time 两种计算）
  - 成功率: 成功/失败数和错误码分布
  - 总数据量: 总传输字节数
