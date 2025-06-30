# reg.docker.alibaba-inc.com/aii/aistudio:7310127-20240506105611
pip install -U openmim
# mim install "mmcv==2.1.0"
ossutil64 cp -r oss://antsys-vilab/zsz/pytorch/torch-1.12.1+cu113-cp310-cp310-linux_x86_64.whl ./
pip install torch-1.12.1+cu113-cp310-cp310-linux_x86_64.whl
ossutil64 cp -r oss://antsys-vilab/zsz/pytorch/torchvision-0.13.1+cu113-cp310-cp310-linux_x86_64.whl ./
pip install torchvision-0.13.1+cu113-cp310-cp310-linux_x86_64.whl
pip install torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
mim install mmengine
ossutil64 cp -r oss://antsys-vilab/zsz/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl ./
pip install mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl
pip install ftfy
pip install regex
pip install plyfile
# pip install -U numpy
# pip install numpy==1.8
# pip install mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl 
pip install "mmsegmentation>=1.0.0"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import torch; print(os.path.dirname(torch.__file__) +"/lib")'`
# ls -all /opt/conda/lib/python3.10/site-packages/torch/lib/libcudnn_ops_infer.so.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.10/site-packages/torch/lib/
echo $LD_LIBRARY_PATH
cp /opt/conda/lib/python3.10/site-packages/torch/lib/libcudnn_ops_infer.so.8 /usr/local/lib
# find / -name "libcudnn_ops_infer.so.8" | xargs -i sh -c 'cp {} /usr/lib/ && echo "Copied: {}"'
# # print all files with name "libcudnn_ops_infer.so.8" and copy them to /usr/lib/
    
#| xargs -i cp {} /usr/lib/
pip install nerfvis