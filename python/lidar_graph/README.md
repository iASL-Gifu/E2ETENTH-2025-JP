#

##
```bash
echo 'export LD_LIBRARY_PATH=/home/tamiya/.local/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


pip install --user -e .
python3 -c "import lidar_graph_cuda; print('OK')"

```