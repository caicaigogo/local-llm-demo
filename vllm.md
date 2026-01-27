不能直接 pip install vllm，因为 PyPI 默认包只包含 CUDA 后端

pip install torch torchvision

git clone https://github.com/vllm-project/vllm.git
git checkout ed6e907

cd vllm
安装用于构建 vLLM CPU 后端的 Python 包

老版本安装 0.7.3
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu


新版本安装
pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu



构建并安装 vLLM CPU 后端
VLLM_TARGET_DEVICE=cpu python setup.py install

想开发 vLLM，请以可编辑模式安装。
VLLM_TARGET_DEVICE=cpu python setup.py develop
