- deepep github链接
	https://github.com/deepseek-ai/DeepEP
- deepep本地源代码位置
	E:\codex_home\code\DeepEP
- 编译服务器
	- （不兼容）L40S服务器信息（支持的cuda版本最高为8.9，deepep要求hopper sm90，不兼容。）
		- 主机名：tf01-l40s-56
		- 内网IP：10.225.16.56
		- 浮动IP：10.220.73.56
		- 连接方式：ssh -i InfraBuild_key zetyun@10.220.73.56
		- key路径：E:\doc\测试gpu_key\InfraBuild_key
		- L40S 是 Ada Lovelace，CUDA compute capability 是 8.9，NVIDIA 官方表中 L40S 位于 8.9 档位；NVIDIA L40S 页面也标明其 FP8/FP16/TF32 能力和 48GB 显存等规格。参考：NVIDIA CUDA GPU 表、NVIDIA L40S 页面。但当前 DeepEP 源码 README 明确要求 Hopper SM90 或支持 SM90 PTX ISA 的架构；本地 setup.py:130 默认 TORCH_CUDA_ARCH_LIST=9.0，而 DISABLE_SM90_FEATURES=1 分支目前直接 assert False，所以这个源码版本对L40S 不是完整受支持目标。
	- （有效）H800服务器信息
		- 名称：gx-test-inst1-0                                 
		- 运行状态：Running
		- ip地址：172.16.84.47     
		- 节点名称：hd04-gpul-0042
	- hd04-gpul-0042连接命令
		- tsh ssh -L 8080:172.16.84.47:8080 --user=guoxu root@hd04-cci-k8s-master-1