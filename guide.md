这个压测脚本的核心设计:
                                                                                                                        
  测试流程:                                                                                                             
  1. Phase 1 - 并发注册: N 个 client 线程同时调用 setup() 向 master 发送 MountSegment RPC 注册，测量注册吞吐和延迟      
  2. Phase 2 - 并发 RPC 压测: 所有 client 通过 threading.Barrier 同步后同时发起 RPC 请求                                
                                                                                                                      
  支持的 RPC 压测类型:                                                                                                  
                                                                                                                      
  ┌──────────────────┬──────────────────────────────────┬─────────────────────┐                                         
  │       类型       │         对应 Master RPC          │        说明         │                                       
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ health_check     │ Ping/ServiceReady                │ 心跳检查            │                                       
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤
  │ is_exist         │ ExistKey                         │ 单 key 存在性查询   │                                         
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ batch_is_exist   │ BatchExistKey                    │ 批量 key 存在性查询 │                                         
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ put_only         │ PutStart + PutEnd                │ 纯写入元数据        │                                       
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ put_get          │ PutStart/PutEnd + GetReplicaList │ 写+读完整链路       │
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ batch_put        │ BatchPutStart/BatchPutEnd        │ 批量写入            │                                       
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ get_replica_desc │ GetReplicaList                   │ 副本信息查询        │                                       
  ├──────────────────┼──────────────────────────────────┼─────────────────────┤                                         
  │ remove           │ Remove                           │ 删除操作            │
  └──────────────────┴──────────────────────────────────┴─────────────────────┘                                         
                                                                                                                      
  使用示例:                                                                                                             
  # 8 个 client 并发，每个做 500 次 RPC
  python stress_master_rpc_benchmark.py \                                                                               
      --master-server localhost:50051 \                                                                                 
      --metadata-server http://127.0.0.1:8080/metadata \
      --num-clients 8 \                                                                                                 
      --ops-per-client 500 \                                                                                          
      --rpc-types health_check,is_exist,put_only                                                                        
                                                                                                                        
  # 16 个 client 全量压测，含详细统计
  python stress_master_rpc_benchmark.py \                                                                               
      --master-server localhost:50051 \                                                                               
      --metadata-server http://127.0.0.1:8080/metadata \                                                                
      --num-clients 16 \                                                                                              
      --ops-per-client 1000 \                                                                                           
      --rpc-types health_check,is_exist,batch_is_exist,put_only,batch_put,get_replica_desc,remove \
      --detailed-stats                                                                                                  
                                                                                                                        
  输出指标: 注册吞吐(clients/s)、每种 RPC 的 p50/p90/p99/p999 延迟、QPS、成功率、错误码统计。
